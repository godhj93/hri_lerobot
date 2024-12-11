import rospy
import numpy as np
import pandas as pd
import time
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PLOT_RESULT = True

X_MIN_BOUND = -0.06
X_MAX_BOUND = 0.06
Y_MIN_BOUND = 0.25
Y_MAX_BOUND = 0.30
Z_MIN_BOUND = 0.051
Z_MAX_BOUND = 0.1

def transform_to_configuration_space(positions):
    """
    데이터를 configuration space로 변환:
    - x는 [-0.2, 0.2] 범위로 정규화
    - y는 [0.1, 0.2] 범위로 정규화
    - z는 [0.1, 0.2] 범위로 정규화
    - x:y 비율은 유지
    """
    transformed_positions = []

    # x와 y의 최소값과 최대값 계산
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    print(x_min, x_max)
    print(y_min, y_max)
    # z의 최소값과 최대값 계산
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    for pos in positions:
        x, y, z = pos

        # x 정규화
        x_normalized = X_MIN_BOUND + (x - x_min) / (x_max - x_min) * (X_MAX_BOUND - (X_MIN_BOUND))

        # y 정규화
        y_normalized = Y_MIN_BOUND + (y - y_min) / (y_max - y_min) * (Y_MAX_BOUND - Y_MIN_BOUND)
        # scale_factor = y_normalized / y if y != 0 else 1.0
        # x_normalized *= scale_factor

        # z 정규화
        z_normalized = Z_MIN_BOUND + (z - z_min) / (z_max - z_min) * (Z_MAX_BOUND - Z_MIN_BOUND)

        # 결과 추가
        transformed_positions.append([x_normalized, y_normalized, z_normalized])

    return np.array(transformed_positions)

def plot_points(points, title):
    """
    3D 플롯
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z, c='blue', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title(title)
    plt.show()


def publish_target_positions(csv_file, topic_name='/target_position', interval=1.0):
    # CSV 파일 로드
    data = pd.read_csv(csv_file)

    # 첫 세 열 추출 (x, y, z)
    positions = data.iloc[:, :3].values

    if PLOT_RESULT: plot_points(positions, 'generated points')

    # normalize to configuration space
    positions = transform_to_configuration_space(positions)

    # plot the end effector
    if PLOT_RESULT: plot_points(positions, 'normalized end effector points')

    # ROS 노드 초기화
    rospy.init_node('csv_target_position_publisher', anonymous=True)
    pub = rospy.Publisher(topic_name, Float32MultiArray, queue_size=10)

    try:
        for position in positions:
            if rospy.is_shutdown():
                break

            # 메시지 생성
            target_msg = Float32MultiArray()
            target_msg.data = position.tolist()

            # 퍼블리시
            pub.publish(target_msg)
            print(f"Published target position: {position}")

            # 주어진 시간 간격 대기
            time.sleep(interval)

    except rospy.ROSInterruptException:
        print("ROS node interrupted. Exiting.")

if __name__ == "__main__":
    # CSV 파일 경로
    csv_file_path = 'test.csv'

    # 퍼블리시 실행
    publish_target_positions(csv_file_path, interval=0.1)