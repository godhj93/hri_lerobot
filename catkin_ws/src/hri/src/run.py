import numpy as np
import mujoco
import mujoco.viewer
from interface import SimulatedRobot
from utils import load_world, fix_joint_angle
import time

# ROS 
import rospy
from std_msgs.msg import Float32MultiArray

def callback(data):
    # target_ee_position = data
    robot.target_ee_position = np.array(data.data)
    print(f"robot.target_ee_position: {robot.target_ee_position}")
    
if __name__ == '__main__':
        
    # Define World, Robot, and Data
    world, data = load_world()
    robot = SimulatedRobot(world, data)

    # Set simulation timestep
    world.opt.timestep = 1.0 / 500  # Simulation timestep in seconds (500 Hz)

    # Initialize the robot position
    # Note: 나중에 로봇에 전원 인가 시 해당 위치로 초기화하는 것 추가해야함 [Not implemented]
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.0
    mujoco.mj_forward(world, data)

    # Desired real-time frame rate
    # Note: 시뮬레이션에서만 필요한 것으로 나중에 로봇 탑재 시 불필요함
    frame_duration = 1.0 / 60  

    # Define target end-effector position
    # Note: ROS를 이용해서 간단한 제어가 가능한지 확인 [Not implemented]
    robot.target_ee_position = np.array([0.0, 0.0, 0.0])  # x, y, z 목표 위치

    # Initialize ROS
    rospy.init_node('robot_interface', anonymous=True)
    rospy.Subscriber('target_position', Float32MultiArray, callback)

    with mujoco.viewer.launch_passive(world, data) as viewer:
        while viewer.is_running():
            
            step_start = time.time()
            
            target_joint_positions = robot.inverse_kinematics_rot_backup(ee_target_pos = robot.target_ee_position, ee_target_rot = fix_joint_angle(), joint_name = 'joint6')
            
            # 관절 위치를 목표로 설정
            robot.set_target_pos(target_joint_positions)

            # 시뮬레이션 한 스텝 전진
            mujoco.mj_step(world, data)

            # 현재 말단 조작기 위치 출력
            current_ee_position = robot.read_ee_pos(joint_name='joint6')
            print()
            if np.linalg.norm(current_ee_position - robot.target_ee_position) < 1e-1:
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
                
            # rospy.spin()
    
    