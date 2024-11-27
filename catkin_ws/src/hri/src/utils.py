import numpy as np
import mujoco
from termcolor import colored
import rospy
from visualization_msgs.msg import Marker

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

def create_marker_traj(ns = "joint6_trajectory"):
    
        trajectory_marker = Marker()
        trajectory_marker.header.frame_id = "world"  # Replace with your Fixed Frame
        trajectory_marker.header.stamp = rospy.Time.now()
        trajectory_marker.ns = ns
        trajectory_marker.id = 0
        trajectory_marker.type = Marker.LINE_STRIP  # Type for trajectory
        trajectory_marker.action = Marker.ADD
        trajectory_marker.scale.x = 0.01  # Line width
        trajectory_marker.color.a = 1.0  # Transparency
        trajectory_marker.color.r = 1.0  # Red
        trajectory_marker.color.g = 0.0
        trajectory_marker.color.b = 0.0
        trajectory_marker.pose.orientation.w = 1.0
        
        return trajectory_marker
    
