import rospy
from std_msgs.msg import Float32MultiArray

def main():
    # ROS 노드 초기화
    rospy.init_node('target_position_publisher', anonymous=True)
    pub = rospy.Publisher('target_position', Float32MultiArray, queue_size=10)

    try:
        while not rospy.is_shutdown():
            print("Enter the target position (x, y, z).")
            try:
                x = float(input("Enter x: "))
                y = float(input("Enter y: "))
                z = float(input("Enter z: "))

                # 메시지 생성
                target_msg = Float32MultiArray()
                target_msg.data = [x, y, z]

                # 퍼블리시
                pub.publish(target_msg)
                print(f"Published target position: x={x}, y={y}, z={z}")

            except ValueError:
                print("Invalid input. Please enter numeric values for x, y, and z.")

    except rospy.ROSInterruptException:
        print("\nROS node interrupted. Exiting program.")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting program.")

if __name__ == "__main__":
    main()