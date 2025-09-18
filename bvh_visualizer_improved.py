import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, Listbox, Checkbutton, IntVar, messagebox
import math
import csv
import colorsys
import sys
sys.path.append('./mocapapi_python') # Add mocapapi_python to sys.path
from mocap_api import MCPApplication, MCPSettings, MCPBvhData, MCPBvhDisplacement, MCPBvhRotation, MCPEventType, MCPAvatar, MCPError, EMCPCommand
from mocap_main_base import MCPBase

# Global MocapAPI instances
mocap_app = None
mocap_settings = None
current_avatar = None

def init_mocap_api():
    global mocap_app, mocap_settings, current_avatar
    print("Initializing MocapAPI...")
    try:
        mocap_app = MCPApplication.create_application()
        mocap_settings = MCPSettings.create_settings()

        # Configure BVH data settings
        mocap_settings.set_bvh_data(MCPBvhData.create_bvh_data(
            rotation=MCPBvhRotation.create_bvh_rotation(True, True, True),
            displacement=MCPBvhDisplacement.create_bvh_displacement(True, True, True)
        ))

        # Configure network settings (example: UDP broadcast)
        # mocap_settings.set_udp_server(MCPUDPServer.create_udp_server("127.0.0.1", 7001))
        # mocap_settings.set_tcp_server(MCPTCPServer.create_tcp_server("127.0.0.1", 7002))

        mocap_app.open(mocap_settings)
        print("MocapAPI initialized and connected successfully.")
    except MCPError as e:
        print(f"Failed to initialize MocapAPI: {e}")
        if mocap_app:
            mocap_app.close()
        mocap_app = None
        mocap_settings = None

try:
    from OpenGL.GLUT import *
except ImportError:
    print("Warning: PyOpenGL-accelerate is not installed. GLUT may not be available.")
    print("Please install PyOpenGL-accelerate: pip install PyOpenGL_accelerate")
    from OpenGL.GLUT import *

# BVH Joint Class
class Joint:
    def __init__(self, name, parent=None):
        self.name = name
        self.children = []
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.channel_indices = {}
        self.matrix = np.identity(4)
        self.end_site = None
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.rom = {'Xrotation': [float('inf'), float('-inf')],
                    'Yrotation': [float('inf'), float('-inf')],
                    'Zrotation': [float('inf'), float('-inf')]}
        self.anatomical_angles = {} 
        self.channel_start_index = 0  # 保存通道起始索引
    
    def add_child(self, child):
        self.children.append(child)
    
    def set_offset(self, offset):
        self.offset = np.array(offset)
    
    def set_channels(self, channels, channel_start_index):
        self.channels = channels
        self.channel_start_index = channel_start_index
        for i, channel in enumerate(channels):
            self.channel_indices[channel] = channel_start_index + i
    
    def set_end_site(self, end_site):
        self.end_site = np.array(end_site)

# BVH File Parser
def parse_bvh(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, {}, [], 0, 0
    lines = [line.strip() for line in lines]
    root_joint = None
    joints = {}
    stack = []
    line_index = 0
    channel_count = 0
    while line_index < len(lines) and lines[line_index] != 'MOTION':
        line = lines[line_index]
        parts = line.split()
        if not parts:
            line_index += 1
            continue
        if parts[0] == 'HIERARCHY':
            line_index += 1
            continue
        elif parts[0] == 'ROOT' or parts[0] == 'JOINT':
            joint_name = parts[1]
            new_joint = Joint(joint_name, parent=stack[-1] if stack else None)
            if not root_joint:
                root_joint = new_joint
            joints[joint_name] = new_joint
            if stack:
                stack[-1].add_child(new_joint)
            stack.append(new_joint)
            line_index += 1
        elif parts[0] == '{':
            line_index += 1
        elif parts[0] == 'OFFSET':
            offset = [float(p) for p in parts[1:]]
            stack[-1].set_offset(offset)
            line_index += 1
        elif parts[0] == 'CHANNELS':
            num_channels = int(parts[1])
            channels = parts[2:]
            stack[-1].set_channels(channels, channel_count)
            channel_count += num_channels
            line_index += 1
        elif parts[0] == 'End' and parts[1] == 'Site':
            line_index += 2
            end_site = [float(p) for p in lines[line_index].split()[1:]]
            stack[-1].set_end_site(end_site)
            line_index += 2
        elif parts[0] == '}':
            if stack:
                stack.pop()
            line_index += 1
        else:
            line_index += 1
    
    motion_data = []
    frames = 0
    frame_time = 0.0
    while line_index < len(lines):
        line = lines[line_index]
        parts = line.split()
        if not parts:
            line_index += 1
            continue
        if parts[0] == 'MOTION':
            line_index += 1
        elif parts[0] == 'Frames:':
            frames = int(parts[1])
            line_index += 1
        elif parts[0] == 'Frame' and parts[1] == 'Time:':
            frame_time = float(parts[2])
            line_index += 1
        else:
            motion_data.append([float(p) for p in parts])
            line_index += 1
    return root_joint, joints, motion_data, frames, frame_time

# 获取关节世界坐标
def get_world_position(joint):
    return joint.matrix[:3, 3]

# 更新关节矩阵
def update_joint_matrices(joint, frame_data, all_joints):
    if joint.parent is None:
        pos_x = frame_data[joint.channel_indices.get('Xposition', -1)] if 'Xposition' in joint.channels else 0
        pos_y = frame_data[joint.channel_indices.get('Yposition', -1)] if 'Yposition' in joint.channels else 0
        pos_z = frame_data[joint.channel_indices.get('Zposition', -1)] if 'Zposition' in joint.channels else 0
        
        T = np.identity(4)
        T[0, 3] = pos_x
        T[1, 3] = pos_y
        T[2, 3] = pos_z
        joint.matrix = T
    else:
        joint.matrix = all_joints[joint.parent.name].matrix @ np.array([
            [1, 0, 0, joint.offset[0]],
            [0, 1, 0, joint.offset[1]],
            [0, 0, 1, joint.offset[2]],
            [0, 0, 0, 1]
        ])
    for channel in joint.channels:
        if 'rotation' in channel:
            axis = channel[0]
            angle = frame_data[joint.channel_indices[channel]]
            
            R = np.identity(4)
            angle_rad = np.radians(angle)
            c = np.cos(angle_rad)
            s = np.sin(angle_rad)
            
            if axis == 'X':
                R = np.array([
                    [1, 0, 0, 0],
                    [0, c, -s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]
                ])
            elif axis == 'Y':
                R = np.array([
                    [c, 0, s, 0],
                    [0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [0, 0, 0, 1]
                ])
            elif axis == 'Z':
                R = np.array([
                    [c, -s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
            
            joint.matrix = joint.matrix @ R
    for child in joint.children:
        update_joint_matrices(child, frame_data, all_joints)

# 计算解剖学角度
def calculate_anatomical_angles(joints):
    angles = {}
    def get_angle(joint_name1, joint_name2, joint_name3):
        if joint_name1 in joints and joint_name2 in joints and joint_name3 in joints:
            p1 = get_world_position(joints[joint_name1])
            p2 = get_world_position(joints[joint_name2])
            p3 = get_world_position(joints[joint_name3])
            
            vec1 = p1 - p2
            vec2 = p3 - p2
            
            if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                return angle
        return None
    Y_AXIS = np.array([0, 1, 0])
    Z_AXIS = np.array([0, 0, 1])
    X_AXIS = np.array([1, 0, 0])
    
    def signed_angle(v1, v2, axis):
        v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 1e-6 else np.array([0,0,0])
        v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 1e-6 else np.array([0,0,0])
        dot_product = np.dot(v1_norm, v2_norm)
        cross_product = np.cross(v1_norm, v2_norm)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        sign = np.sign(np.dot(cross_product, axis))
        if sign == 0:
            sign = 1
        return np.degrees(angle_rad) * sign
    # Upper Body
    if 'RightArm' in joints and 'RightShoulder' in joints:
        arm_vec = get_world_position(joints['RightArm']) - get_world_position(joints['RightShoulder'])
        yz_plane_vec = np.array([0, arm_vec[1], arm_vec[2]])
        angles['RightShoulder_Abduction/Adduction'] = signed_angle(Y_AXIS, yz_plane_vec, X_AXIS)
        xy_plane_vec = np.array([arm_vec[0], arm_vec[1], 0])
        angles['RightShoulder_Flexion/Extension'] = signed_angle(Y_AXIS, xy_plane_vec, Z_AXIS)
    
    if 'LeftArm' in joints and 'LeftShoulder' in joints:
        arm_vec = get_world_position(joints['LeftArm']) - get_world_position(joints['LeftShoulder'])
        yz_plane_vec = np.array([0, arm_vec[1], arm_vec[2]])
        angles['LeftShoulder_Abduction/Adduction'] = signed_angle(Y_AXIS, yz_plane_vec, -X_AXIS)
        xy_plane_vec = np.array([arm_vec[0], arm_vec[1], 0])
        angles['LeftShoulder_Flexion/Extension'] = signed_angle(Y_AXIS, xy_plane_vec, -Z_AXIS)
    # Elbow Flexion
    angles['RightElbow_Flexion'] = get_angle('RightArm', 'RightForeArm', 'RightHand')
    angles['LeftElbow_Flexion'] = get_angle('LeftArm', 'LeftForeArm', 'LeftHand')
    
    # Lower Body
    if 'RightUpLeg' in joints and 'Hips' in joints:
        leg_vec = get_world_position(joints['RightUpLeg']) - get_world_position(joints['Hips'])
        yz_plane_vec = np.array([0, leg_vec[1], leg_vec[2]])
        angles['RightHip_Flexion/Extension'] = signed_angle(Y_AXIS, yz_plane_vec, -X_AXIS)
        xy_plane_vec = np.array([leg_vec[0], leg_vec[1], 0])
        angles['RightHip_Abduction/Adduction'] = signed_angle(Y_AXIS, xy_plane_vec, -Z_AXIS)
    if 'LeftUpLeg' in joints and 'Hips' in joints:
        leg_vec = get_world_position(joints['LeftUpLeg']) - get_world_position(joints['Hips'])
        yz_plane_vec = np.array([0, leg_vec[1], leg_vec[2]])
        angles['LeftHip_Flexion/Extension'] = signed_angle(Y_AXIS, yz_plane_vec, X_AXIS)
        xy_plane_vec = np.array([leg_vec[0], leg_vec[1], 0])
        angles['LeftHip_Abduction/Adduction'] = signed_angle(Y_AXIS, xy_plane_vec, Z_AXIS)
    # Knee Flexion
    angles['RightKnee_Flexion'] = get_angle('RightUpLeg', 'RightLeg', 'RightFoot')
    angles['LeftKnee_Flexion'] = get_angle('LeftUpLeg', 'LeftLeg', 'LeftFoot')
    return {key: val for key, val in angles.items() if val is not None}

# 计算运动学数据
def calculate_kinematics(joints, all_frames_data, frame_time):
    num_frames = len(all_frames_data)
    
    positions_per_frame = []
    anatomical_angles_per_frame = []
    
    for frame_data in all_frames_data:
        temp_joints = {}
        for name, joint in joints.items():
            temp_joints[name] = Joint(joint.name, parent=joint.parent)
            temp_joints[name].set_offset(joint.offset)
            temp_joints[name].set_channels(joint.channels, joint.channel_start_index)
            if joint.end_site is not None:
                temp_joints[name].set_end_site(joint.end_site)
        for name, joint in joints.items():
            for child in joint.children:
                if child.name in temp_joints:
                    temp_joints[name].add_child(temp_joints[child.name])
        temp_root = temp_joints[list(joints.keys())[0]]
        
        update_joint_matrices(temp_root, frame_data, temp_joints)
        
        frame_positions = {name: get_world_position(joint) for name, joint in temp_joints.items()}
        positions_per_frame.append(frame_positions)
        
        frame_anatomical_angles = calculate_anatomical_angles(temp_joints)
        anatomical_angles_per_frame.append(frame_anatomical_angles)
    velocities_per_frame = []
    accelerations_per_frame = []
    
    for i in range(num_frames):
        current_velocities = {}
        current_accelerations = {}
        
        for name in joints:
            if i == 0 or i == 1:
                current_velocities[name] = np.zeros(3)
                current_accelerations[name] = np.zeros(3)
            else:
                pos_diff = positions_per_frame[i][name] - positions_per_frame[i-1][name]
                velocity = pos_diff / frame_time
                current_velocities[name] = velocity
                
                vel_diff = current_velocities[name] - velocities_per_frame[i-1][name]
                acceleration = vel_diff / frame_time
                current_accelerations[name] = acceleration
        velocities_per_frame.append(current_velocities)
        accelerations_per_frame.append(current_accelerations)
    return positions_per_frame, velocities_per_frame, accelerations_per_frame, anatomical_angles_per_frame

# 计算关节活动度
def calculate_rom(all_anatomical_angles):
    rom_data = {}
    
    if not all_anatomical_angles:
        return rom_data
    if all_anatomical_angles[0]:
        for key in all_anatomical_angles[0]:
            rom_data[key] = [float('inf'), float('-inf')]
    else:
        return rom_data
        
    for frame_angles in all_anatomical_angles:
        for joint_key, angle_val in frame_angles.items():
            if joint_key in rom_data:
                if angle_val < rom_data[joint_key][0]:
                    rom_data[joint_key][0] = angle_val
                if angle_val > rom_data[joint_key][1]:
                    rom_data[joint_key][1] = angle_val
                
    return rom_data

# 自定义骨骼渲染顺序
CUSTOM_JOINT_ORDER = [
    'Hips',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    'Spine', 'Spine1', 'Spine2',
    'Neck', 'Neck1', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3',
    'RightinHandindex', 'RightHandindex1', 'RightHandindex2', 'RightHandindex3',
    'RightlnHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3',
    'RightinHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3',
    'RightinHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',
    'LeftinHandindex', 'LeftHandindex1', 'LeftHandindex2', 'LeftHandindex3',
    'LeftinHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
    'LeftinHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3',
    'LeftinHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',
    'Spine3'
]


# 从MocapAPI的Avatar数据更新BVH关节
def update_joint_from_avatar(bvh_joints, avatar):
    for joint_name, bvh_joint in bvh_joints.items():
        mocap_joint = avatar.get_joint_by_name(joint_name)
        if mocap_joint:
            # 更新关节的局部旋转
            w, x, y, z = mocap_joint.get_local_rotation()
            bvh_joint.rotation = np.array([w, x, y, z])

            # 更新关节的局部位置（如果适用，通常只有根关节有位移）
            if bvh_joint.parent is None:  # 根关节
                pos_x, pos_y, pos_z = mocap_joint.get_local_position()
                bvh_joint.position = np.array([pos_x, pos_y, pos_z])

            # 重新计算关节的全局变换矩阵
            bvh_joint.update_matrix()

# 重构骨骼渲染函数
def draw_skeleton_custom_order(joints):
    for joint_name in CUSTOM_JOINT_ORDER:
        if joint_name not in joints:
            continue
        joint = joints[joint_name]
        
        # 渲染当前关节（球体）
        glColor3f(0.0, 0.0, 0.0)
        glPushMatrix()
        joint_pos = joint.matrix[:3, 3]
        glTranslatef(joint_pos[0], joint_pos[1], joint_pos[2])
        quad = gluNewQuadric()
        gluSphere(quad, 2.5 * 0.4, 16, 16)
        gluDeleteQuadric(quad)
        glPopMatrix()
        
        # 绘制当前关节与父关节的连接线条
        if joint.parent is not None and joint.parent.name in joints:
            parent_joint = joints[joint.parent.name]
            parent_pos = parent_joint.matrix[:3, 3]
            
            glLineWidth(2.0)
            glColor3f(0.0, 0.0, 0.0)
            glBegin(GL_LINES)
            glVertex3f(*parent_pos)
            glVertex3f(*joint_pos)
            glEnd()
        
        # 绘制End Site（如手指末端）
        if joint.end_site is not None:
            end_site_pos = joint.matrix @ np.append(joint.end_site, 1.0)
            end_site_pos = end_site_pos[:3]
            
            glLineWidth(2.0)
            glColor3f(0.0, 0.0, 0.0)
            glBegin(GL_LINES)
            glVertex3f(*joint_pos)
            glVertex3f(*end_site_pos)
            glEnd()
            
            glPushMatrix()
            glTranslatef(end_site_pos[0], end_site_pos[1], end_site_pos[2])
            quad = gluNewQuadric()
            gluSphere(quad, 2.5 * 0.3, 16, 16)
            gluDeleteQuadric(quad)
            glPopMatrix()

# 绘制直角矩形
def draw_rect(x, y, width, height, color):
    glColor3f(*color)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()

# 左侧Position面板
def draw_position_panel(display, current_positions, joints):
    panel_x = 10
    panel_y = display[1] - 60  # 位于按钮下方
    line_height = 18
    title_font = GLUT_BITMAP_HELVETICA_18
    content_font = GLUT_BITMAP_HELVETICA_12
    title_color = (0.0, 0.0, 0.0)
    content_color = (0.0, 0.0, 0.0)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], 0, display[1], -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    # 计算数据列的总宽度
    joint_name_col_start = panel_x + 10
    X_COL_START = joint_name_col_start + 120
    Y_COL_START = X_COL_START + 62
    Z_COL_START = Y_COL_START + 60
    data_column_width = Z_COL_START - joint_name_col_start + 60  # 假设Z列数据宽度为60，可根据实际调整

    # 计算标题的水平居中位置
    title_text = "All Joints - Position (m)"
    title_width = len(title_text) * 8  # 假设每个字符宽度为8，根据实际字体调整
    title_x = joint_name_col_start + (data_column_width - title_width) // 2

    # 绘制面板标题
    draw_text_2d(title_x, panel_y, title_text, title_color, title_font)
    current_y = panel_y - line_height

    # 按自定义顺序遍历关节
    for joint_name in CUSTOM_JOINT_ORDER:
        if joint_name not in joints or joint_name not in current_positions:
            continue
        # 绘制关节名称
        draw_text_2d(joint_name_col_start, current_y, joint_name, content_color, content_font)
        # cm转m，保留4位小数
        pos = current_positions[joint_name] / 100
        x_text = f"X:{pos[0]:.4f}"
        y_text = f"Y:{pos[1]:.4f}"
        z_text = f"Z:{pos[2]:.4f}"
        draw_text_2d(X_COL_START, current_y, x_text, content_color, content_font)
        draw_text_2d(Y_COL_START, current_y, y_text, content_color, content_font)
        draw_text_2d(Z_COL_START, current_y, z_text, content_color, content_font)
        current_y -= line_height
        if current_y < 50:  # 底部留50像素
            break

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

# 右侧Velocity面板
# 右侧Velocity面板（修复参数错误，与左侧面板对齐）
def draw_velocity_panel(display, current_velocities, joints):
    panel_x = display[0] - 330  # 右侧预留330像素宽度
    panel_y = display[1] - 60  # 与Position面板顶部对齐
    line_height = 18
    title_font = GLUT_BITMAP_HELVETICA_18
    content_font = GLUT_BITMAP_HELVETICA_12
    title_color = (0.0, 0.0, 0.0)
    content_color = (0.0, 0.0, 0.0)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], 0, display[1], -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    # 计算数据列的总宽度
    joint_name_col_start = panel_x + 10
    X_COL_START = joint_name_col_start + 120
    Y_COL_START = X_COL_START + 62
    Z_COL_START = Y_COL_START + 60
    data_column_width = Z_COL_START - joint_name_col_start + 60  # 假设Z列数据宽度为60，可根据实际调整

    # 计算标题的水平居中位置
    title_text = "All Joints - Velocity (m/s)"
    title_width = len(title_text) * 8  # 假设每个字符宽度为8，根据实际字体调整
    title_x = joint_name_col_start + (data_column_width - title_width) // 2

    # 绘制面板标题
    draw_text_2d(title_x, panel_y, title_text, title_color, title_font)
    current_y = panel_y - line_height

    # 按自定义顺序遍历关节
    for joint_name in CUSTOM_JOINT_ORDER:
        if joint_name not in joints or joint_name not in current_velocities:
            continue
        # 绘制关节名称
        draw_text_2d(joint_name_col_start, current_y, joint_name, content_color, content_font)
        # cm/s转m/s，保留4位小数
        vel = current_velocities[joint_name] / 100
        x_text = f"X:{vel[0]:.4f}"
        y_text = f"Y:{vel[1]:.4f}"
        z_text = f"Z:{vel[2]:.4f}"
        draw_text_2d(X_COL_START, current_y, x_text, content_color, content_font)
        draw_text_2d(Y_COL_START, current_y, y_text, content_color, content_font)
        draw_text_2d(Z_COL_START, current_y, z_text, content_color, content_font)
        current_y -= line_height
        if current_y < 50:  # 底部留50像素
            break

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

# 绘制坐标轴及标签
def draw_axes_and_labels():
    modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    axis_length = 16.67
    label_offset = 21.0
    glColor3f(1.0, 0.0, 0.0)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(axis_length, 0.0, 0.0)
    glEnd()
    try:
        x_pos_3d = gluProject(label_offset, 0, 0, modelview_matrix, projection_matrix, viewport)
        glWindowPos2d(x_pos_3d[0], x_pos_3d[1])
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord('X'))
    except ValueError:
        pass
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, axis_length, 0.0)
    glEnd()
    try:
        y_pos_3d = gluProject(0, label_offset, 0, modelview_matrix, projection_matrix, viewport)
        glWindowPos2d(y_pos_3d[0], y_pos_3d[1])
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord('Y'))
    except ValueError:
        pass
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, axis_length)
    glEnd()
    try:
        z_pos_3d = gluProject(0, 0, label_offset, modelview_matrix, projection_matrix, viewport)
        glWindowPos2d(z_pos_3d[0], z_pos_3d[1])
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord('Z'))
    except ValueError:
        pass

# 绘制网格
def draw_grid():
    glPushMatrix()
    glLineWidth(1.0)
    glColor3f(0.8, 0.8, 0.8)
    
    glBegin(GL_LINES)
    for i in range(-10, 11):
        glVertex3f(i * 50, 0, -500)
        glVertex3f(i * 50, 0, 500)
        glVertex3f(-500, 0, i * 50)
        glVertex3f(500, 0, i * 50)
    glEnd()
    glPopMatrix()

# 绘制2D文本
def draw_text_2d(x, y, text, color, font=GLUT_BITMAP_HELVETICA_18):
    glColor3f(*color)
    glWindowPos2d(x, y)
    for char in text:
        glutBitmapCharacter(font, ord(char))

# -------------------------- 优化：关节轨迹绘制函数（实线→绿色小点，仅播放时显示） --------------------------
def draw_joint_trajectories(show_trajectories, selected_joints, joint_trajectories, joint_colors, current_frame):
    if not show_trajectories or not selected_joints or not joint_trajectories:
        return
    
    glDisable(GL_DEPTH_TEST)  # 轨迹在骨骼上方显示
    glColor3f(0.0, 1.0, 0.0)  # 固定绿色
    glPointSize(1.0)          # 点大小（3像素，精致不突兀）
    
    for joint_name in selected_joints:
        if joint_name not in joint_trajectories:
            continue
        trajectory = joint_trajectories[joint_name]
        if len(trajectory) < 1:
            continue
        
        # 仅绘制当前帧及之前的点（随播放进度累积）
        glBegin(GL_POINTS)
        for i in range(0, current_frame + 1):
            if i >= len(trajectory):
                break
            pos = trajectory[i]
            glVertex3f(*pos)
        glEnd()
    
    glEnable(GL_DEPTH_TEST)

# -------------------------- 轨迹设置窗口（关节多选+开关） --------------------------
def open_trajectory_settings(joints, all_joint_positions, show_trajectories, selected_joints, joint_trajectories, joint_colors):
    if not joints or not all_joint_positions:
        tk.Tk().withdraw()
        tk.messagebox.showwarning("提示", "请先加载BVH文件！")
        return
    
    # 新建Tkinter窗口
    settings_win = tk.Tk()
    settings_win.title("关节轨迹设置")
    settings_win.geometry("300x400")
    
    # 轨迹总开关
    show_var = IntVar(value=1 if show_trajectories else 0)
    show_checkbox = Checkbutton(
        settings_win,
        text="显示关节轨迹",
        variable=show_var,
        font=("Arial", 10)
    )
    show_checkbox.pack(pady=10, anchor="w", padx=20)
    
    # 关节多选列表
    tk.Label(settings_win, text="选择关节（可多选）：", font=("Arial", 10)).pack(anchor="w", padx=20)
    listbox = Listbox(
        settings_win,
        selectmode=tk.MULTIPLE,  # 支持多选
        font=("Arial", 9),
        height=15
    )
    # 加载所有关节名（按自定义顺序）
    joint_names = [name for name in CUSTOM_JOINT_ORDER if name in joints]
    for idx, name in enumerate(joint_names):
        listbox.insert(idx, name)
        # 已选中的关节默认勾选
        if name in selected_joints:
            listbox.selection_set(idx)
    listbox.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
    
    # 确认按钮逻辑
    def confirm_settings():
        nonlocal show_trajectories, selected_joints, joint_trajectories, joint_colors
        
        # 1. 更新轨迹开关状态
        show_trajectories = bool(show_var.get())
        
        # 2. 更新选中的关节
        selected_indices = listbox.curselection()
        selected_joints = [joint_names[idx] for idx in selected_indices]
        
        # 3. 构建选中关节的轨迹数据（从all_joint_positions提取）
        joint_trajectories.clear()
        for joint_name in selected_joints:
            trajectory = []
            for frame_pos in all_joint_positions:
                # 提取该关节在当前帧的位置
                pos = frame_pos.get(joint_name, np.zeros(3))
                trajectory.append(pos)
            joint_trajectories[joint_name] = trajectory
        
        # 4. 为每个选中关节分配专属颜色（HSV色轮，区分明显）
        joint_colors.clear()
        num_joints = len(selected_joints)
        for idx, joint_name in enumerate(selected_joints):
            hue = idx / num_joints if num_joints > 0 else 0  # 色调均匀分布
            saturation = 0.7  # 饱和度
            value = 0.8  # 明度
            # HSV转RGB（简化计算）
            color = colorsys.hsv_to_rgb(hue, saturation, value)
            joint_colors[joint_name] = color
        
        settings_win.destroy()
    
    # 确认按钮
    confirm_btn = tk.Button(
        settings_win,
        text="确认",
        command=confirm_settings,
        font=("Arial", 10),
        width=10
    )
    confirm_btn.pack(pady=10)
    
    settings_win.mainloop()
    # 返回更新后的数据（用于主函数变量同步）
    return show_trajectories, selected_joints, joint_trajectories, joint_colors

# 绘制2D UI
def draw_2d_ui(display, current_frame, frames, is_playing, fps, load_btn_rect, export_btn_rect, trajectory_btn_rect, play_pause_btn_rect, timeline_rect, bvh_fps=0, bvh_total_frames=0):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], 0, display[1], -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    
    # 1. Load按钮
    load_x = load_btn_rect.x
    load_y = display[1] - load_btn_rect.y - load_btn_rect.height
    load_width = load_btn_rect.width
    load_height = load_btn_rect.height
    draw_rect(load_x, load_y, load_width, load_height, (0.8, 0.8, 0.8))
    glColor3f(0.0, 0.0, 0.0)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(load_x, load_y)
    glVertex2f(load_x + load_width, load_y)
    glVertex2f(load_x + load_width, load_y + load_height)
    glVertex2f(load_x, load_y + load_height)
    glEnd()
    load_text = "Load File"
    text_width = len(load_text) * 8
    text_height = 12
    load_text_x = load_x + (load_width - text_width) / 2 + 8  
    load_text_y = load_y + (load_height + text_height) / 2 - 10  
    draw_text_2d(load_text_x, load_text_y, load_text, (0.0, 0.0, 0.0), font=GLUT_BITMAP_HELVETICA_12)
    
    # 2. Export按钮
    export_x = export_btn_rect.x
    export_y = display[1] - export_btn_rect.y - export_btn_rect.height
    export_width = export_btn_rect.width
    export_height = export_btn_rect.height
    draw_rect(export_x, export_y, export_width, export_height, (0.8, 0.8, 0.8))
    glColor3f(0.0, 0.0, 0.0)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(export_x, export_y)
    glVertex2f(export_x + export_width, export_y)
    glVertex2f(export_x + export_width, export_y + export_height)
    glVertex2f(export_x, export_y + export_height)
    glEnd()
    export_text = "Export Data"
    export_text_width = len(export_text) * 8
    export_text_height = 12
    export_text_x = export_x + (export_width - export_text_width) / 2 + 8  
    export_text_y = export_y + (export_height + export_text_height) / 2 - 10 
    draw_text_2d(export_text_x, export_text_y, export_text, (0.0, 0.0, 0.0), font=GLUT_BITMAP_HELVETICA_12)
    
    # 3. 轨迹设置按钮
    traj_x = trajectory_btn_rect.x
    traj_y = display[1] - trajectory_btn_rect.y - trajectory_btn_rect.height
    traj_width = trajectory_btn_rect.width
    traj_height = trajectory_btn_rect.height
    draw_rect(traj_x, traj_y, traj_width, traj_height, (0.8, 0.8, 0.8))
    glColor3f(0.0, 0.0, 0.0)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(traj_x, traj_y)
    glVertex2f(traj_x + traj_width, traj_y)
    glVertex2f(traj_x + traj_width, traj_y + traj_height)
    glVertex2f(traj_x, traj_y + traj_height)
    glEnd()
    traj_text = "Trajectory"
    traj_text_width = len(traj_text) * 8
    traj_text_height = 12
    traj_text_x = traj_x + (traj_width - traj_text_width) / 2 + 8  
    traj_text_y = traj_y + (traj_height + traj_text_height) / 2 - 10 
    draw_text_2d(traj_text_x, traj_text_y, traj_text, (0.0, 0.0, 0.0), font=GLUT_BITMAP_HELVETICA_12)
    
    # 时间轴相关绘制
    if frames > 0:
        draw_text_2d(timeline_rect.x - 10, timeline_rect.y, "0", (0.0, 0.0, 0.0), font=GLUT_BITMAP_HELVETICA_12)
        draw_text_2d(timeline_rect.x + timeline_rect.width + 10, timeline_rect.y, str(frames - 1), (0.0, 0.0, 0.0), font=GLUT_BITMAP_HELVETICA_12)
        frame_text = f"Frame: {current_frame}"
        draw_text_2d((display[0] - len(frame_text)*8) // 2, timeline_rect.y + timeline_rect.height + 5, frame_text, (0.0, 0.0, 0.0), font=GLUT_BITMAP_HELVETICA_12)
        
        glColor3f(0.7, 0.7, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(timeline_rect.x, timeline_rect.y)
        glVertex2f(timeline_rect.x + timeline_rect.width, timeline_rect.y)
        glVertex2f(timeline_rect.x + timeline_rect.width, timeline_rect.y + timeline_rect.height)
        glVertex2f(timeline_rect.x, timeline_rect.y + timeline_rect.height)
        glEnd()
        progress_width = (current_frame / (frames - 1)) * timeline_rect.width if frames > 1 else 0
        glColor3f(0.4, 0.4, 0.4)
        glBegin(GL_QUADS)
        glVertex2f(timeline_rect.x, timeline_rect.y)
        glVertex2f(timeline_rect.x + progress_width, timeline_rect.y)
        glVertex2f(timeline_rect.x + progress_width, timeline_rect.y + timeline_rect.height)
        glVertex2f(timeline_rect.x, timeline_rect.y + timeline_rect.height)
        glEnd()
        
        slider_x = timeline_rect.x + progress_width
        slider_y = timeline_rect.y
        slider_w = 8
        slider_h = 16
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_QUADS)
        glVertex2f(slider_x - slider_w/2, slider_y - slider_h/2 + timeline_rect.height/2)
        glVertex2f(slider_x + slider_w/2, slider_y - slider_h/2 + timeline_rect.height/2)
        glVertex2f(slider_x + slider_w/2, slider_y + slider_h/2 + timeline_rect.height/2)
        glVertex2f(slider_x - slider_w/2, slider_y + slider_h/2 + timeline_rect.height/2)
        glEnd()
    
    # 播放/暂停按钮
    glColor3f(0.0, 0.0, 0.0)
    if is_playing:
        glBegin(GL_QUADS)
        glVertex2f(play_pause_btn_rect.x, play_pause_btn_rect.y)
        glVertex2f(play_pause_btn_rect.x + play_pause_btn_rect.width * 0.4, play_pause_btn_rect.y)
        glVertex2f(play_pause_btn_rect.x + play_pause_btn_rect.width * 0.4, play_pause_btn_rect.y + play_pause_btn_rect.height)
        glVertex2f(play_pause_btn_rect.x, play_pause_btn_rect.y + play_pause_btn_rect.height)
        glEnd()
        glBegin(GL_QUADS)
        glVertex2f(play_pause_btn_rect.x + play_pause_btn_rect.width * 0.6, play_pause_btn_rect.y)
        glVertex2f(play_pause_btn_rect.x + play_pause_btn_rect.width, play_pause_btn_rect.y)
        glVertex2f(play_pause_btn_rect.x + play_pause_btn_rect.width, play_pause_btn_rect.y + play_pause_btn_rect.height)
        glVertex2f(play_pause_btn_rect.x + play_pause_btn_rect.width * 0.6, play_pause_btn_rect.y + play_pause_btn_rect.height)
        glEnd()
    else:
        glBegin(GL_TRIANGLES)
        glVertex2f(play_pause_btn_rect.x, play_pause_btn_rect.y)
        glVertex2f(play_pause_btn_rect.x, play_pause_btn_rect.y + play_pause_btn_rect.height)
        glVertex2f(play_pause_btn_rect.x + play_pause_btn_rect.width, play_pause_btn_rect.y + play_pause_btn_rect.height/2)
        glEnd()
    
    # BVH数据信息显示
    if bvh_fps > 0 and bvh_total_frames > 0:
        bvh_info_text = f"BVH Data: {bvh_fps:.0f}HZ, {bvh_total_frames - 1}Frames"
        draw_text_2d(10, 30, bvh_info_text, (0.0, 0.0, 0.0), font=GLUT_BITMAP_HELVETICA_12)
    # 软件帧率显示
    fps_text = f"BVH Viewer: {int(fps)} FPS"
    draw_text_2d(10, 10, fps_text, (0.0, 0.0, 0.0), font=GLUT_BITMAP_HELVETICA_12)
    
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

# 绘制关节角度标签
def draw_joint_angle_label(joint1_name, joint2_name, joint3_name, joints, display, arc_radius=3.3, color=(0.5, 0.5, 0.5)):
    if joint1_name not in joints or joint2_name not in joints:
        return
    p1 = joints[joint1_name].matrix[:3, 3]
    p2 = joints[joint2_name].matrix[:3, 3]
    
    if joint3_name in joints:
        p3 = joints[joint3_name].matrix[:3, 3]
    else:
        if joints[joint2_name].end_site is not None and len(joints[joint2_name].end_site) == 3:
            end_site_pos = joints[joint2_name].matrix @ np.append(joints[joint2_name].end_site, 1.0)
            p3 = end_site_pos[:3]
        else:
            return
    vec1 = p1 - p2
    vec2 = p3 - p2
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    angle_rad = np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    axis = np.cross(vec1_norm, vec2_norm)
    if np.linalg.norm(axis) == 0:
        return
    axis = axis / np.linalg.norm(axis)
    start_vector = vec1_norm * arc_radius
    angle_step = 5
    glLineWidth(1.5)
    glColor3f(*color)
    glBegin(GL_LINE_STRIP)
    for i in range(0, int(angle_deg) + 1, angle_step):
        angle_current_rad = np.radians(i)
        
        rotated_vector = start_vector * np.cos(angle_current_rad) + \
                         np.cross(axis, start_vector) * np.sin(angle_current_rad) + \
                         axis * np.dot(axis, start_vector) * (1 - np.cos(angle_current_rad))
        
        arc_point = p2 + rotated_vector
        glVertex3f(*arc_point)
    glEnd()
    text_pos_3d = p2 + (vec1_norm + vec2_norm) / 2.0 * arc_radius * 1.5
    modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    try:
        text_pos_2d = gluProject(text_pos_3d[0], text_pos_3d[1], text_pos_3d[2], modelview_matrix, projection_matrix, viewport)
        glWindowPos2d(text_pos_2d[0], text_pos_2d[1])
        glColor3f(0.0, 0.0, 0.0)
        angle_text = f"{angle_deg:.1f}°"
        for char in angle_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
    except ValueError:
        pass

# 反投影
def unproject(winX, winY, winZ=0.0):
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    obj_point = gluUnProject(winX, winY, winZ, modelview, projection, viewport)
    return obj_point

# 导出数据
def export_data_dialog(all_joints, all_positions, all_velocities, all_accelerations, all_anatomical_angles):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    root.destroy()
    
    if not file_path:
        print("数据导出已取消。")
        return
    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            header = ['Frame']
            # 按自定义顺序导出关节数据
            joint_names = [name for name in CUSTOM_JOINT_ORDER if name in all_joints]
            angle_keys = list(all_anatomical_angles[0].keys()) if all_anatomical_angles else []
            for joint_name in joint_names:
                header.append(f'{joint_name}_pos_X')
                header.append(f'{joint_name}_pos_Y')
                header.append(f'{joint_name}_pos_Z')
                header.append(f'{joint_name}_vel_X')
                header.append(f'{joint_name}_vel_Y')
                header.append(f'{joint_name}_vel_Z')
                header.append(f'{joint_name}_accel_X')
                header.append(f'{joint_name}_accel_Y')
                header.append(f'{joint_name}_accel_Z')
            
            for angle_key in angle_keys:
                header.append(angle_key)
            writer.writerow(header)
            
            num_frames = len(all_positions)
            for i in range(num_frames):
                row = [i + 1]
                
                for joint_name in joint_names:
                    # 单位转换：cm→m，cm/s→m/s，cm/s²→m/s²
                    pos = all_positions[i].get(joint_name, np.zeros(3)) / 100
                    vel = all_velocities[i].get(joint_name, np.zeros(3)) / 100
                    accel = all_accelerations[i].get(joint_name, np.zeros(3)) / 100
                    
                    row.extend(pos)
                    row.extend(vel)
                    row.extend(accel)
                
                for angle_key in angle_keys:
                    angle_val = all_anatomical_angles[i].get(angle_key, 'N/A')
                    row.append(angle_val)
                
                writer.writerow(row)
                
        print(f"数据成功导出到 {file_path}")
    except Exception as e:
        print(f"数据导出失败: {e}")

# 主函数
def main(mocap_client):
    pygame.init()
    glutInit()
    
    # 1. 改为可调整窗口（原代码保留）
    display = (1280, 960)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | pygame.RESIZABLE)  # 用变量screen接收窗口对象
    pygame.display.set_caption("BVH 3D Viewer")  # 原标题栏文字
    init_mocap_api()

    
    # -------------------------- 新增：设置标题栏/缩略图 Logo --------------------------
    try:
        # 加载 ICO 图标文件（需与脚本同目录）
        app_icon = pygame.image.load("app_icon.ico")
        # 设置窗口图标（同时生效于标题栏和任务栏缩略图）
        pygame.display.set_icon(app_icon)
    except Exception as e:
        print(f"加载标题栏图标失败：{e}（请确保 app_icon.ico 在脚本目录下）")
    # --------------------------------------------------------------------------------
    
    glClearColor(1.0, 1.0, 1.0, 1.0) 
    glEnable(GL_DEPTH_TEST)
    
    # 后续原代码不变...
    
    glClearColor(1.0, 1.0, 1.0, 1.0) 
    glEnable(GL_DEPTH_TEST)
    
    # 新增：存储实时窗口宽高比（用于拉伸时更新投影）
    aspect_ratio = display[0] / display[1]
    
    root_joint, joints, motion_data, frames, frame_time = None, {}, [], 0, 0
    current_frame = 0
    is_playing = False
    
    # BVH数据的帧率（HZ）和总帧数
    bvh_fps = 0.0
    bvh_total_frames = 0
    
    target_fps = 60
    clock = pygame.time.Clock()
    left_button_down = False
    middle_button_down = False
    timeline_dragging = False
    
    last_mouse_pos = (0, 0)
    
    # 轨迹相关变量
    show_trajectories = False  # 轨迹总开关（默认关闭）
    selected_joints = []       # 选中的关节列表
    joint_trajectories = {}    # 关节轨迹数据：{关节名: [帧1位置, 帧2位置, ...]}
    joint_colors = {}          # 关节颜色映射：{关节名: (R, G, B)}
    
    # 按钮矩形定义（Load→Export→Trajectory）
    btn_y = 10
    btn_height = 25
    load_btn_rect = pygame.Rect(10, btn_y, 90, btn_height)
    export_btn_rect = pygame.Rect(
        load_btn_rect.x + load_btn_rect.width + 10, 
        btn_y, 
        110, 
        btn_height
    )
    trajectory_btn_rect = pygame.Rect(
        export_btn_rect.x + export_btn_rect.width + 10,
        btn_y,
        110,
        btn_height
    )

    # MocapAPI 控制按钮
    connect_mocap_btn_rect = pygame.Rect(
        trajectory_btn_rect.x + trajectory_btn_rect.width + 30,
        btn_y,
        120,
        btn_height
    )
    disconnect_mocap_btn_rect = pygame.Rect(
        connect_mocap_btn_rect.x + connect_mocap_btn_rect.width + 10,
        btn_y,
        120,
        btn_height
    )
    start_stream_btn_rect = pygame.Rect(
        disconnect_mocap_btn_rect.x + disconnect_mocap_btn_rect.width + 30,
        btn_y,
        120,
        btn_height
    )
    stop_stream_btn_rect = pygame.Rect(
        start_stream_btn_rect.x + start_stream_btn_rect.width + 10,
        btn_y,
        120,
        btn_height
    )
    
    play_pause_btn_rect = pygame.Rect(0, 0, 0, 0) 
    timeline_rect = pygame.Rect(0, 0, 0, 0)
    
    all_joint_positions = []
    all_joint_velocities = []
    all_joint_accelerations = []
    all_anatomical_angles = []
    joint_roms = {}

    def reset_view():
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    # 2. 用实时aspect_ratio替代固定宽高比，支持窗口拉伸
        gluPerspective(45, aspect_ratio, 0.1, 1000.0)
        glTranslatef(0.0, -100.0, -300)
        
    def toggle_play_pause():
        nonlocal is_playing
        is_playing = not is_playing
    
    def load_file_dialog():
        nonlocal root_joint, joints, motion_data, frames, frame_time, current_frame, all_joint_positions, all_joint_velocities, all_joint_accelerations, all_anatomical_angles, joint_roms, bvh_fps, bvh_total_frames
        # 如果 MocapAPI 正在运行，则禁用文件加载
        if mocap_app and mocap_app.is_open():
            messagebox.showinfo("MocapAPI 运行中", "MocapAPI 正在运行，无法加载 BVH 文件。")
            return

        # 加载新文件时清空旧轨迹数据
        nonlocal show_trajectories, selected_joints, joint_trajectories, joint_colors
        show_trajectories = False
        selected_joints.clear()
        joint_trajectories.clear()
        joint_colors.clear()
        
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(defaultextension=".bvh", filetypes=[("BVH files", "*.bvh")])
        root.destroy()
        
        if file_path:
            root_joint, joints, motion_data, frames, frame_time = parse_bvh(file_path)
            if root_joint:
                motion_data = np.array(motion_data)
                frames = len(motion_data)
                current_frame = 0
                
                bvh_total_frames = frames
                if frame_time > 0:
                    bvh_fps = 1.0 / frame_time
                else:
                    bvh_fps = 0.0
                
                print(f"成功加载文件: {file_path}")
                print(f"BVH Data: {bvh_fps:.0f}HZ, {bvh_total_frames - 1}Frames")
                print(f"帧时间: {frame_time}s")
                
                all_joint_positions, all_joint_velocities, all_joint_accelerations, all_anatomical_angles = calculate_kinematics(joints, motion_data, frame_time)
                joint_roms = calculate_rom(all_anatomical_angles)
                print("运动学数据和关节活动度计算完成。")
            else:
                print(f"文件解析失败: {file_path}")
                bvh_fps = 0.0
                bvh_total_frames = 0
    
    reset_view()
    running = True
    while running:
        # 播放按钮位置更新
        play_btn_size = 20
        play_btn_x = (display[0] - play_btn_size) // 2
        play_btn_y = display[1] - 30 - play_btn_size
        play_pause_btn_rect.update(play_btn_x, play_btn_y, play_btn_size, play_btn_size)
        
        # 时间轴位置更新
        timeline_width = display[0] - 200
        timeline_x = (display[0] - timeline_width) // 2
        timeline_height = 8
        timeline_y = play_btn_y - 10 - timeline_height
        timeline_rect.update(timeline_x, timeline_y, timeline_width, timeline_height)
        
        # 事件处理（优化鼠标操作逻辑）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                mocap_api.cleanup()
                pygame.quit()
                quit()

            # MocapAPI 事件处理
            if mocap_app:
                mocap_event = mocap_app.poll_event()
                if mocap_event and mocap_event.event_type == MCPEventType.AvatarUpdated:
                    global current_avatar
                    current_avatar = mocap_event.event_data.avatar_updated.avatar
                    # print(f"Avatar Updated: {current_avatar.get_avatar_index()}")

            # 窗口拉伸事件（修复黑屏+移除GLUT函数，避免闪退）
            elif event.type == pygame.VIDEORESIZE:
                # 更新窗口尺寸和宽高比
                display = (event.w, event.h)
                aspect_ratio = event.w / event.h  # 实时更新宽高比
                
                # 重建窗口：保留双缓冲+硬件加速，避免缓冲区清空
                pygame.display.set_mode(
                    display, 
                    DOUBLEBUF | OPENGL | pygame.RESIZABLE | pygame.HWSURFACE  # 硬件加速减少黑屏
                )
                
                # 重置视图+强制即时重绘（用OpenGL原生命令替代glutPostRedisplay）
                reset_view()
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 清空旧缓冲
                glFlush()  # 强制OpenGL执行清空命令
                # 绘制 MocapAPI 控制按钮
        pygame.draw.rect(screen, (200, 200, 200), connect_mocap_btn_rect)
        draw_text_2d(connect_mocap_btn_rect.x + 5, connect_mocap_btn_rect.y + 5, "连接 MocapAPI", (0, 0, 0))

        pygame.draw.rect(screen, (200, 200, 200), disconnect_mocap_btn_rect)
        draw_text_2d(disconnect_mocap_btn_rect.x + 5, disconnect_mocap_btn_rect.y + 5, "断开 MocapAPI", (0, 0, 0))

        pygame.draw.rect(screen, (200, 200, 200), start_stream_btn_rect)
        draw_text_2d(start_stream_btn_rect.x + 5, start_stream_btn_rect.y + 5, "开始数据流", (0, 0, 0))

        pygame.draw.rect(screen, (200, 200, 200), stop_stream_btn_rect)
        draw_text_2d(stop_stream_btn_rect.x + 5, stop_stream_btn_rect.y + 5, "停止数据流", (0, 0, 0))

        # 绘制 MocapAPI 控制按钮
        pygame.draw.rect(screen, (200, 200, 200), connect_mocap_btn_rect)
        draw_text_2d(connect_mocap_btn_rect.x + 5, connect_mocap_btn_rect.y + 5, "连接 MocapAPI", (0, 0, 0))

        pygame.draw.rect(screen, (200, 200, 200), disconnect_mocap_btn_rect)
        draw_text_2d(disconnect_mocap_btn_rect.x + 5, disconnect_mocap_btn_rect.y + 5, "断开 MocapAPI", (0, 0, 0))

        pygame.draw.rect(screen, (200, 200, 200), start_stream_btn_rect)
        draw_text_2d(start_stream_btn_rect.x + 5, start_stream_btn_rect.y + 5, "开始数据流", (0, 0, 0))

        pygame.draw.rect(screen, (200, 200, 200), stop_stream_btn_rect)
        draw_text_2d(stop_stream_btn_rect.x + 5, stop_stream_btn_rect.y + 5, "停止数据流", (0, 0, 0))

        pygame.display.flip()  # 立即刷新Pygame窗口，避免黑屏
                
        # 同步更新UI控件位置（避免缩放后UI错位）
        play_btn_x = (display[0] - play_btn_size) // 2
        play_btn_y = display[1] - 30 - play_btn_size
        play_pause_btn_rect.update(play_btn_x, play_btn_y, play_btn_size, play_btn_size)
        
        timeline_width = display[0] - 200

        # 更新 MocapAPI 控制按钮位置
        connect_mocap_btn_rect.update(
            trajectory_btn_rect.x + trajectory_btn_rect.width + 30,
            btn_y,
            120,
            btn_height
        )
        disconnect_mocap_btn_rect.update(
            connect_mocap_btn_rect.x + connect_mocap_btn_rect.width + 10,
            btn_y,
            120,
            btn_height
        )
        start_stream_btn_rect.update(
            disconnect_mocap_btn_rect.x + disconnect_mocap_btn_rect.width + 30,
            btn_y,
            120,
            btn_height
        )
        stop_stream_btn_rect.update(
            start_stream_btn_rect.x + start_stream_btn_rect.width + 10,
            btn_y,
            120,
            btn_height
        )
        timeline_x = (display[0] - timeline_width) // 2
        timeline_y = play_btn_y - 10 - timeline_height
        timeline_rect.update(timeline_x, timeline_y, timeline_width, timeline_height)
        # 鼠标按下事件（左键平移、中键按下恢复、右键旋转、滚轮基于视角缩放）
        if event.type == pygame.MOUSEBUTTONDOWN:
            last_mouse_pos = event.pos
            
            if event.button == 1:
                left_button_down = True  # 左键平移状态
            elif event.button == 2:
                reset_view()  # 中键按下恢复初始视图
            elif event.button == 3:
                middle_button_down = True  # 右键旋转状态
            elif event.button == 4:
                # 滚轮上滚：基于当前视角放大（靠近画面中心）
                view_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
                # 提取相机朝向（视图矩阵第3列，负方向为相机正前方）
                cam_forward = np.array([-view_matrix[2][0], -view_matrix[2][1], -view_matrix[2][2]])
                cam_forward = cam_forward / np.linalg.norm(cam_forward)  # 归一化方向
                glTranslatef(*(cam_forward * 10.0))  # 沿朝向移动（放大）
            elif event.button == 5:
                # 滚轮下滚：基于当前视角缩小（远离画面中心）
                view_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
                cam_forward = np.array([-view_matrix[2][0], -view_matrix[2][1], -view_matrix[2][2]])
                cam_forward = cam_forward / np.linalg.norm(cam_forward)
                glTranslatef(*(cam_forward * -10.0))  # 逆朝向移动（缩小）
                
                # 按钮点击逻辑（Load/Export等）不变...
                if event.button == 1:
                    if load_btn_rect.collidepoint(event.pos):
                        load_file_dialog()
                    elif export_btn_rect.collidepoint(event.pos) and frames > 0:
                        export_data_dialog(joints, all_joint_positions, all_joint_velocities, all_joint_accelerations, all_anatomical_angles)
                    elif trajectory_btn_rect.collidepoint(event.pos):
                        updated_vals = open_trajectory_settings(
                            joints, all_joint_positions,
                            show_trajectories, selected_joints,
                            joint_trajectories, joint_colors
                        )
                        if updated_vals:
                            show_trajectories, selected_joints, joint_trajectories, joint_colors = updated_vals
                    elif connect_mocap_btn_rect.collidepoint(event.pos):
                        print("连接 MocapAPI 按钮被点击")
                        mocap_client.open()
                    elif disconnect_mocap_btn_rect.collidepoint(event.pos):
                        print("断开 MocapAPI 按钮被点击")
                        mocap_client.close()
                    elif start_stream_btn_rect.collidepoint(event.pos):
                        print("开始数据流按钮被点击")
                        mocap_client.running_command(EMCPCommand.CommandStartCapture)
                    elif stop_stream_btn_rect.collidepoint(event.pos):
                        print("停止数据流按钮被点击")
                        mocap_client.running_command(EMCPCommand.CommandStopCapture)
                    elif play_pause_btn_rect.collidepoint(event.pos):
                        toggle_play_pause()
                    elif timeline_rect.collidepoint(event.pos) and frames > 0:
                        timeline_dragging = True
                        is_playing = False
                        current_frame = int((event.pos[0] - timeline_rect.x) / timeline_rect.width * (frames - 1))
            # 鼠标松开事件
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    # 左键松开：停止平移（原有逻辑不变）
                    left_button_down = False
                    timeline_dragging = False
                elif event.button == 3:
                    # 右键松开：停止旋转（原中键松开逻辑）
                    middle_button_down = False  # 与按下时的状态变量保持一致
            
            # 鼠标移动事件（左键平移、右键绕中心旋转、时间轴拖动）
            # 鼠标移动事件（左键平移、右键绕火柴人水平旋转、时间轴拖动）
            if event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                rel_x, rel_y = mouse_x - last_mouse_pos[0], mouse_y - last_mouse_pos[1]
                
                # 左键平移（原有逻辑不变，保留）
                if left_button_down and not timeline_dragging:
                    view_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
                    right_axis = np.array([view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]])
                    up_axis = np.array([view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]])
                    translate_x = rel_x * 0.2 * right_axis
                    translate_y = -rel_y * 0.2 * up_axis
                    glTranslatef(translate_x[0], translate_y[1], translate_x[2] + translate_y[2])
                
                # 右键拖动：仅水平绕火柴人（Hips关节）旋转（简化逻辑）
                if middle_button_down and joints:  # 确保已加载关节数据
                    try:
                        # 1. 获取火柴人根关节（Hips，骨盆）的世界坐标（旋转中心）
                        # 若Hips不存在， fallback到第一个关节（兼容不同BVH骨骼命名）
                        target_joint = joints.get('Hips', joints[next(iter(joints.keys()))])
                        joint_world_pos = target_joint.matrix[:3, 3]  # Hips的世界位置
                        
                        # 2. 旋转逻辑：先移到Hips中心→水平旋转→移回原位置
                        glTranslatef(*joint_world_pos)  # 把Hips移到世界原点（旋转中心）
                        # 仅水平旋转（绕Y轴，左右拖动有效，上下拖动无效），速度0.15更平缓
                        glRotatef(rel_x * 0.15, 0, 1, 0)  # 只响应鼠标X轴偏移（左右拖）
                        glTranslatef(-joint_world_pos[0], -joint_world_pos[1], -joint_world_pos[2])  # 移回原位
                    except Exception as e:
                        print(f"绕火柴人旋转异常: {e}")
                        pass
                
                # 时间轴拖动（原有逻辑不变，保留）
                if timeline_dragging and frames > 0:
                    mouse_pos_x = event.pos[0]
                    progress_x = min(max(mouse_pos_x, timeline_rect.x), timeline_rect.right)
                    current_frame = int((progress_x - timeline_rect.x) / timeline_rect.width * (frames - 1))
                
                last_mouse_pos = (mouse_x, mouse_y)
            
            # 键盘事件
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    toggle_play_pause()  # 原有空格播放/暂停逻辑不变
                if event.key == pygame.K_LEFT and frames > 0:
                    current_frame = max(0, current_frame - 1)  # 原有左键帧后退
                if event.key == pygame.K_RIGHT and frames > 0:
                    current_frame = min(frames - 1, current_frame + 1)  # 原有右键帧前进
                # 新增：F键触发恢复初始视图
                if event.key == pygame.K_f:
                    reset_view()
        
        # 渲染流程（仅播放时显示轨迹点）
        # 渲染流程（优化缩放时实时性）
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glFlush()  # 新增：强制清空缓冲命令即时执行，避免延迟导致黑屏
        glMatrixMode(GL_MODELVIEW)
        
        glPushMatrix()
        draw_grid()
        draw_axes_and_labels()
        
        if root_joint and len(motion_data) > 0:
            if current_avatar:
                update_joint_from_avatar(joints, current_avatar)
            elif is_playing:
                update_joint_matrices(root_joint, motion_data[current_frame], joints)
                current_frame = (current_frame + 1) % frames
            else:
                update_joint_matrices(root_joint, motion_data[current_frame], joints)
            
            draw_skeleton_custom_order(joints)  # 渲染骨骼（原有逻辑不变）
            # 仅播放时绘制轨迹点（传入当前帧控制打点范围）
            if is_playing:
                draw_joint_trajectories(
                    show_trajectories,
                    selected_joints,
                    joint_trajectories,
                    joint_colors,
                    current_frame
                )
            
            # 绘制关节角度标签
            draw_joint_angle_label('RightShoulder', 'RightArm', 'RightForeArm', joints, display, arc_radius=3.3, color=(0.9, 0.2, 0.2))
            draw_joint_angle_label('RightArm', 'RightForeArm', 'RightHand', joints, display, arc_radius=3.3, color=(0.9, 0.2, 0.2))
            draw_joint_angle_label('LeftShoulder', 'LeftArm', 'LeftForeArm', joints, display, arc_radius=3.3, color=(0.2, 0.2, 0.9))
            draw_joint_angle_label('LeftArm', 'LeftForeArm', 'LeftHand', joints, display, arc_radius=3.3, color=(0.2, 0.2, 0.9))
            draw_joint_angle_label('Hips', 'RightUpLeg', 'RightLeg', joints, display, arc_radius=5.0, color=(0.9, 0.2, 0.2))
            draw_joint_angle_label('RightUpLeg', 'RightLeg', 'RightFoot', joints, display, arc_radius=5.0, color=(0.9, 0.2, 0.2))
            draw_joint_angle_label('Hips', 'LeftUpLeg', 'LeftLeg', joints, display, arc_radius=5.0, color=(0.2, 0.2, 0.9))
            draw_joint_angle_label('LeftUpLeg', 'LeftLeg', 'LeftFoot', joints, display, arc_radius=5.0, color=(0.2, 0.2, 0.9))
            
        glPopMatrix()
        fps = clock.get_fps() if clock.get_fps() > 0 else 0
        
        # 调整UI坐标
        play_btn_y_bottom_up = 30
        timeline_y_bottom_up = play_btn_y_bottom_up + play_btn_size + 10
        timeline_rect_opengl = pygame.Rect(timeline_rect.x, timeline_y_bottom_up, timeline_rect.width, timeline_rect.height)
        play_pause_btn_rect_opengl = pygame.Rect(play_pause_btn_rect.x, play_btn_y_bottom_up, play_pause_btn_rect.width, play_pause_btn_rect.height)
        
        # 绘制2D UI
        draw_2d_ui(
            display, 
            current_frame, 
            frames, 
            is_playing, 
            fps, 
            load_btn_rect,   # Load按钮
            export_btn_rect, # Export按钮
            trajectory_btn_rect,  # 轨迹设置按钮
            play_pause_btn_rect_opengl, 
            timeline_rect_opengl,
            bvh_fps=bvh_fps,
            bvh_total_frames=bvh_total_frames
        )
        
        # 绘制Position和Velocity面板
        if all_joint_positions and all_joint_velocities and frames > 0:
            if 0 <= current_frame < len(all_joint_positions) and 0 <= current_frame < len(all_joint_velocities):
                current_positions = all_joint_positions[current_frame]
                current_velocities = all_joint_velocities[current_frame]
                draw_position_panel(display, current_positions, joints)
                draw_velocity_panel(display, current_velocities, joints)
        
        pygame.display.flip()
        
        if is_playing and frame_time > 0:
            target_fps = 1.0 / frame_time
        
        clock.tick(target_fps)

if __name__ == '__main__':
    main(MCPBase())