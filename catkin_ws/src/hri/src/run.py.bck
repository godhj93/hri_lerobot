import numpy as np
import mujoco
import mujoco.viewer
from interface import SimulatedRobot
from utils import load_world, fix_joint_angle, create_marker_traj
import time

# ROS 
import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

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

    # Desired real-time frame rate
    # Note: 시뮬레이션에서만 필요한 것으로 나중에 로봇 탑재 시 불필요함
    frame_duration = 1.0 / 60  

    # Define target end-effector position
    # Note: ROS를 이용해서 간단한 제어가 가능한지 확인 [O]
    x_goal1 = np.linspace(0.0, 0.0, 100)
    y_goal1 = np.linspace(0.1, 0.4, 100)
    z_goal1 = np.linspace(0.1, 0.1, 100)
    
    x_goal2 = np.linspace(0.0, 0.0, 100)
    y_goal2 = np.linspace(0.4, 0.4, 100)
    z_goal2 = np.linspace(0.1, 0.1, 100)
    
    x_goal3 = np.linspace(0.0, 0.3, 100)
    y_goal3 = np.linspace(0.2, 0.2, 100)
    z_goal3 = np.linspace(0.1, 0.1, 100)
    
    x_goal4 = np.linspace(0.3, 0.0, 100)
    y_goal4 = np.linspace(0.2, 0.2, 100)
    z_goal4 = np.linspace(0.1, 0.1, 100)
    
    x_goal5 = np.linspace(0.0, -0.3, 100)
    y_goal5 = np.linspace(0.2, 0.2, 100)
    z_goal5 = np.linspace(0.1, 0.1, 100)
    
    x_goal6 = np.linspace(-0.3, 0.0, 100)
    y_goal6 = np.linspace(0.2, 0.2, 100)
    z_goal6 = np.linspace(0.1, 0.1, 100)
    
    x_goal = np.concatenate((x_goal1, x_goal2, x_goal3, x_goal4, x_goal5))
    y_goal = np.concatenate((y_goal1, y_goal2, y_goal3, y_goal4, y_goal5))
    z_goal = np.concatenate((z_goal1, z_goal2, z_goal3, z_goal4, z_goal5))
    
    i=0
    data.qpos[:] = np.array([x_goal[i], y_goal[i], z_goal[i], 0, 0, 0])
    
    mujoco.mj_forward(world, data)
    
    robot.target_ee_position = np.array([x_goal[i], y_goal[i], z_goal[i]])
    print(robot.target_ee_position )
    # Initialize ROS
    rospy.init_node('robot_interface', anonymous=True)
    rospy.Subscriber('target_position', Float32MultiArray, callback)
    # Publisher for joint6 position visualization
    marker_pub = rospy.Publisher('joint6_trajectory', Marker, queue_size=10)

    # Create and Initialize Marker for trajectory
    trajectory_marker = create_marker_traj()
    
    with mujoco.viewer.launch_passive(world, data) as viewer:
        while viewer.is_running():
            
            step_start = time.time()
            
            target_joint_positions = robot.inverse_kinematics_rot_backup(
                ee_target_pos = robot.target_ee_position, 
                ee_target_rot = fix_joint_angle(), 
                joint_name = 'joint6')
            
            
            if target_joint_positions is not None:
            #     # Simple PID Contoller
            #     kp = 1.0
                
            #     # read current joint positions
            #     current_joint_positions = robot.read_ee_pos(joint_name='joint6')
                
            #     # calculate error
            #     print(f"Target: {target_joint_positions}, Current: {current_joint_positions}")
            #     error = target_joint_positions - current_joint_positions
                
            #     # calculate control signal
            #     control_signal = kp * error
                
            #     # set control signal
            #     target_joint_positions = current_joint_positions + control_signal
                
                # 관절 위치를 목표로 설정
                robot.set_target_pos(target_joint_positions)

            # 시뮬레이션 한 스텝 전진
            mujoco.mj_step(world, data)

            # Read the ee position to visualize in RViz
            current_ee_position = robot.read_ee_pos(joint_name='joint6')
            
            point = Point()
            point.x = current_ee_position[0]
            point.y = current_ee_position[1]
            point.z = current_ee_position[2]
            trajectory_marker.points.append(point)
            marker_pub.publish(trajectory_marker)
            
            if np.linalg.norm(current_ee_position - robot.target_ee_position) < 1e-1:
                i += 1
                robot.target_ee_position = np.array([x_goal[i], y_goal[i], z_goal[i]])
                print("Target reached!")
                # time.sleep(1)
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
    
    