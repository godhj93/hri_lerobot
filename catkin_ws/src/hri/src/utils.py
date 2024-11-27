import numpy as np
import mujoco
from termcolor import colored

def load_world(world_path = 'low_cost_robot/scene.xml'):
        
    # Load world and data
    world = mujoco.MjModel.from_xml_path(world_path)
    data = mujoco.MjData(world)
    print(colored(f"World is loaded from {world_path}", 'green'))
    return world, data
    
def calculate_target_rotation():
    z_axis = np.array([0, 0, 1], dtype=np.float64)  # Gripper z-axis aligned with world z-axis
    x_axis = np.array([1, 0, 0], dtype=np.float64)  # Arbitrary x-axis direction (adjustable)
    
    # Ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    # Construct rotation matrix
    ee_target_rot = np.column_stack((x_axis, y_axis, z_axis))  # 3x3 rotation matrix
    
    return ee_target_rot

def rotation_matrix_to_quaternion(rot_matrix):
    
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_matrix.flatten())
    return quat

def fix_joint_angle():
    
    rot = calculate_target_rotation()
    return rot.flatten()

