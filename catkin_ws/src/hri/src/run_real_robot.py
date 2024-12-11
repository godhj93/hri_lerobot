import numpy as np
import mujoco
import mujoco.viewer
from interface import SimulatedRobot
from utils import load_world, fix_joint_angle, create_marker_traj
import time
import signal  # Ctrl+C 신호 처리
from queue import Queue  # 목표 위치 큐 사용

# ROS
import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from termcolor import colored

from robot import Robot
from utils import radian2pwm, clock
# for callback to use robot.target_ee_position
robot = None

# printing frequency
printing_freq = 500
last_printed_time = 0

# Global variable to handle shutdown
shutdown_flag = False

# Queue for target positions
target_queue = Queue()
prev_target_reached = True
prev_reached_time = 0
not_reached_duration = 0
let_it_go_thre = 1.0

DRAWING_Z = 0.07
DRAWING_POINT_DIST = 0.001
INTERPOLATING_NUM = 10
def signal_handler(signum, frame):
    global shutdown_flag
    shutdown_flag = True
    print("Shutdown signal received. Exiting...")

'''
def callback(data):
    global target_queue
    target_queue.put(np.array(data.data))  # 새로운 목표를 큐에 추가
    print(f"New target added to queue: {data.data}")
'''

def callback(data):
    
    # robot.target_ee_position = np.array(data.data)
    '''Original Code'''
    global target_queue, robot
    
    # 새로 받은 데이터
    new_point = np.array(data.data)

    # 큐의 마지막 포인트 가져오기
    if not target_queue.empty():
        last_point = target_queue.queue[-1]
    else:
        # if robot == None: return
        try:
            last_point = robot.target_ee_position
        except:
            return

    # 두 점 사이의 거리 계산
    distance = np.linalg.norm(new_point - last_point)

    if distance > DRAWING_POINT_DIST:
        # 두 점 사이를 짧은 간격으로 보간
        num_interpolated_points = int(np.ceil(distance / DRAWING_POINT_DIST))
        print(num_interpolated_points)
        interpolated_points = np.linspace(last_point, new_point, num_interpolated_points, endpoint=False)
        # 보간된 점들을 큐에 추가
        for point in interpolated_points[1:]:  # 첫 점은 이미 큐에 있으므로 제외
            target_queue.put(point)
            print(f"Interpolated point added to queue: {point}")

    target_queue.put(new_point)
    print(f"New target added to queue: {new_point}")


if __name__ == '__main__':
    last_printed_time = time.time()

    # Set up signal handling for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    print("Hello, Mujoco!")

    # Define World, Robot, and Data
    world, data = load_world()
    robot = SimulatedRobot(world, data)

    # Set simulation timestep
    world.opt.timestep = 1.0 / 500  # Simulation timestep in seconds (500 Hz)

    # Initialize the robot position
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.0
    mujoco.mj_forward(world, data)

    # Desired real-time frame rate
    frame_duration = 1.0 / 60  

    # Initialize ROS
    rospy.init_node('robot_interface', anonymous=True)
    rospy.Subscriber('target_position', Float32MultiArray, callback)
    joint_6_marker_pub = rospy.Publisher('joint6_trajectory', Marker, queue_size=10)
    drawing_marker_pub_4 = rospy.Publisher('drawing_trajectory_4', Marker, queue_size=10)
    drawing_marker_pub_5 = rospy.Publisher('drawing_trajectory_5', Marker, queue_size=10)
    drawing_marker_pub_6 = rospy.Publisher('drawing_trajectory_6', Marker, queue_size=10)
    target_trajectory_pub = rospy.Publisher('target_trajectory', Marker, queue_size=10)
    # Create and Initialize Marker for trajectory
    target_trajectory_marker = create_marker_traj()
    trajectory_marker = create_marker_traj()
    drawing_marker4 = create_marker_traj()
    drawing_marker5 = create_marker_traj()
    drawing_marker6 = create_marker_traj()

    # Initialize target position
    robot.target_ee_position = np.array([0.0, 0.3, 0.1])  # 초기 목표 위치

    trajectory = {
        'x': [],
        'y': [],
        'z': []
    }
    
    real_robot = Robot(device_name='/dev/ttyACM0')
    real_robot._enable_torque()
    real_robot._set_position_control()
    initialized_flag = False
    
    last_debug_time = time.time()
    with mujoco.viewer.launch_passive(world, data) as viewer:
        try:
            while viewer.is_running() and not shutdown_flag:
                step_start = time.time()

                # Check if there is a new target in the queue
                if (not target_queue.empty()) and prev_target_reached:
                    prev_target_reached = False
                    robot.target_ee_position = target_queue.get()  # 큐에서 새로운 목표 가져오기
                    print(f"Processing new target: {robot.target_ee_position}")
                    # a = input("OK?")

                # Calculate the joint positions for the target position
                
                target_radian = robot.inverse_kinematics_rot_backup_5DOF(
                    ee_target_pos=robot.target_ee_position, 
                    ee_target_rot=fix_joint_angle(), 
                    joint_name='joint5')
                
                current_position = np.array(real_robot.read_position())

                # smooth_mover = np.linspace(current_position, radian2pwm(target_radian)[:5], 100)
                if initialized_flag == False: 
                    smooth_mover = np.linspace(current_position, radian2pwm(target_radian)[:5], 100)
                    print("initalized")
                    initialized_flag = True
                else:
                    if robot.target_ee_position[-1] > DRAWING_Z * 1.001:
                        smooth_mover = np.linspace(current_position, radian2pwm(target_radian)[:5], 100)
                        z_up_flag = True
                        print(colored(f"robot pos: {robot.target_ee_position[-1]}, smooth mover : {2000}", 'green'))
                    elif z_up_flag:
                        smooth_mover = np.linspace(current_position, radian2pwm(target_radian)[:5], 1000)
                        z_up_flag = False
                        print(colored(f"smooth mover : {1000}", 'blue'))
                    else:
                        smooth_mover = np.linspace(current_position, radian2pwm(target_radian)[:5], INTERPOLATING_NUM)
                        print(colored(f"smooth mover : {INTERPOLATING_NUM}", 'red'))
                
                step_start = time.time()
                for pwm in smooth_mover:
                    real_robot.set_goal_pos([int(p) for p in pwm])
                    step_start = clock(step_start, world)
                
                    # Update the simulation with the real robot's joint positions
                    current_position = np.array(real_robot.read_position())
                    data.qpos[:5] = robot._pwm2pos(current_position)
                    # 로봇에 대한 관절각도 radian을 업데이트함
                    mujoco.mj_step(robot.m, data)
                    viewer.sync()
                    
                print(colored(f"Finsihed moving to {radian2pwm(target_radian)[:5]}", 'red'))
                if time.time() - last_debug_time > 1:
                    last_debug_time = time.time()
                    
                    print(f"Current position: {current_position}")
                    print(f"Target position: {radian2pwm(target_radian)}")
                    print(f"Target Reached: {prev_target_reached}")
                    
                    # real_robot._disable_torque()
                
                
                # real_robot.set_goal_pos(dynamixel_positions)
                
                # Read the end-effector position to visualize in RViz
                current_ee_position4 = robot.read_ee_pos(joint_name='joint4')
                current_ee_position5 = robot.read_ee_pos(joint_name='joint5')
                # current_ee_position6 = robot.read_ee_pos(joint_name='joint6')
                
                point4 = Point()
                point4.x = current_ee_position4[0]
                point4.y = current_ee_position4[1]
                point4.z = current_ee_position4[2]
                
                point5 = Point()
                point5.x = current_ee_position5[0]
                point5.y = current_ee_position5[1]
                point5.z = current_ee_position5[2]
                
                # point6 = Point()
                # point6.x = current_ee_position6[0]
                # point6.y = current_ee_position6[1]
                # point6.z = current_ee_position6[2]
                target_pts = Point()
                target_pts.x = robot.target_ee_position[0]
                target_pts.y = robot.target_ee_position[1]
                target_pts.z = robot.target_ee_position[2]
                if point4.z < DRAWING_Z:
                    
                    # print(colored("Drawing trajectory 4. Skipped it.", 'yellow'))
                    drawing_marker4.points.append(point4)
                    drawing_marker4.scale.x = 0.001
                    drawing_marker4.scale.y = 0.001
                    drawing_marker4.scale.z = 0.001
                    drawing_marker4.color.r = 1.0
                    drawing_marker4.color.g = 0.0
                    drawing_marker4.color.b = 0.0
                    drawing_marker4.color.a = 1.0
                    drawing_marker_pub_4.publish(drawing_marker4)
                    
                if point5.z < DRAWING_Z:
                    
                    # print(colored("Drawing trajectory 5.", 'blue'))
                    drawing_marker5.points.append(point5)
                    drawing_marker5.scale.x = 0.001
                    drawing_marker5.scale.y = 0.001
                    drawing_marker5.scale.z = 0.001
                    drawing_marker5.color.r = 0.0
                    drawing_marker5.color.g = 1.0
                    drawing_marker5.color.b = 0.0
                    drawing_marker5.color.a = 1.0
                    drawing_marker_pub_5.publish(drawing_marker5)
                    
                # if point6.z < DRAWING_Z:
                    
                #     # print(colored("Drawing trajectory 6", 'green'))
                #     drawing_marker6.points.append(point6)
                #     drawing_marker6.scale.x = 0.001
                #     drawing_marker6.scale.y = 0.001
                #     drawing_marker6.scale.z = 0.001
                #     drawing_marker6.color.r = 0.0
                #     drawing_marker6.color.g = 0.0
                #     drawing_marker6.color.b = 1.0
                #     drawing_marker6.color.a = 1.0
                #     drawing_marker_pub_6.publish(drawing_marker6)

                if target_pts.z < DRAWING_Z:
                    target_trajectory_marker.points.append(target_pts)
                    target_trajectory_marker.scale.x = 0.001
                    target_trajectory_marker.scale.y = 0.001
                    target_trajectory_marker.scale.z = 0.001
                    target_trajectory_marker.color.r = 1.0
                    target_trajectory_marker.color.g = 1.0
                    target_trajectory_marker.color.b = 1.0
                    target_trajectory_marker.color.a = 1.0
                    target_trajectory_pub.publish(target_trajectory_marker)
                    
                # Print out current status
                if time.time() > last_printed_time + 1/printing_freq:
                    last_printed_time = time.time()
                    # rospy.loginfo(f"Position Error: {np.linalg.norm(current_ee_position5 - robot.target_ee_position, ord=2)}"), 
                    # rospy.loginfo(f"{current_ee_position5 - robot.target_ee_position}")
                    # print(f"Target ee position:  [{np.round(robot.target_ee_position, 2)}]")
                    # print(f"Current ee position: [{np.round(point6.x, 2)}, {np.round(point6.y, 2)}, {np.round(point6.z)}]")
                if np.linalg.norm(current_ee_position5 - robot.target_ee_position) < 5e-2:
                    prev_target_reached = True
                    prev_reached_time = time.time()
                    # rospy.loginfo("Target reached...!")

                else:
                    # print("Target NOT reached...!")
                    not_reached_duration = time.time() - prev_reached_time
                    prev_target_reached = False
                    # print("not reached duration: ", not_reached_duration)
                    if not_reached_duration > let_it_go_thre:
                        not_reached_duration = 0.0
                        prev_reached_time = time.time()
                        prev_target_reached = True
                        
                        # print(f"Cannot reach [{np.round(robot.target_ee_position, 2)}]. Skipped it.")
                # print(f"Current target queue size: {target_queue.qsize()}")
                # print()

                # Synchronize with the viewer
                # viewer.sync()

                # Calculate elapsed time for this step
                elapsed_time = time.time() - step_start
                time_until_next_frame = frame_duration - elapsed_time

                # Sleep to maintain the real-time frame rate
                if time_until_next_frame > 0:
                    time.sleep(time_until_next_frame)
                    
        except rospy.ROSInterruptException:
            print("ROS interrupt received.")
        finally:
            print("Exiting program.")