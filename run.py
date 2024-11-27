import time
import numpy as np
import mujoco
import mujoco.viewer

from interface import SimulatedRobot

# Load model and data
model = mujoco.MjModel.from_xml_path('low_cost_robot/scene.xml')
data = mujoco.MjData(model)

robot = SimulatedRobot(model, data)

# Set simulation timestep
model.opt.timestep = 1.0 / 500  # Simulation timestep in seconds (500 Hz)

# Initialize the robot position
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 0.0
mujoco.mj_forward(model, data)

# Desired real-time frame rate (e.g., 60 FPS)
frame_duration = 1.0 / 60  

target_ee_position = np.array([0.2, 0.1, 0.1])  # x, y, z 목표 위치



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
    
    print(f"Target Rotation Matrix:\n{ee_target_rot}")
    return ee_target_rot

def rotation_matrix_to_quaternion(rot_matrix):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_matrix.flatten())
    return quat

def fix_joint_angle():
    
    rot = calculate_target_rotation()
    # rot = rotation_matrix_to_quaternion(rot)
    # print(f'Quat: {rot}')
    return rot.flatten()

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        target_joint_positions = robot.inverse_kinematics_rot_backup(ee_target_pos = target_ee_position, ee_target_rot = fix_joint_angle(), joint_name = 'joint6')
        
        # 관절 위치를 목표로 설정
        robot.set_target_pos(target_joint_positions)

        # 시뮬레이션 한 스텝 전진
        mujoco.mj_step(model, data)

        # 현재 말단 조작기 위치 출력
        current_ee_position = robot.read_ee_pos(joint_name='joint6')
        print(f"Current EE Position: {current_ee_position}")
        
        if np.linalg.norm(current_ee_position - target_ee_position) < 1e-1:
            print("Target reached!")
            # break
        
        # Synchronize with the viewer
        viewer.sync()

        # Calculate elapsed time for this step
        elapsed_time = time.time() - step_start
        time_until_next_frame = frame_duration - elapsed_time

        # Sleep to maintain the real-time frame rate
        if time_until_next_frame > 0:
            time.sleep(time_until_next_frame)
