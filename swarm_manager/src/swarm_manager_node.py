#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu

class SwarmManagerNode(Node):
    def __init__(self):
        super().__init__('swarm_manager_node')
        self.robot_names = ['robot_1', 'robot_2', 'robot_3', 'robot_4', 'robot_5', 'robot_6']
        self.robot_data = {
            'robot_1': {"odom": None, "scan": None, 'imu': None},
            'robot_2': {"odom": None, "scan": None, 'imu': None},
            'robot_3': {"odom": None, "scan": None, 'imu': None},
            'robot_4': {"odom": None, "scan": None, 'imu': None},
            'robot_5': {"odom": None, "scan": None, 'imu': None},
            'robot_6': {"odom": None, "scan": None, 'imu': None},
        }
        self.robot_subscriptions = {}
        for robot in self.robot_names:
            self.robot_subscriptions[robot] = {
                "odom_sub": self.create_subscription(
                    Odometry,
                    f'/{robot}/odom',
                    lambda msg, r=robot: self.odom_callback(msg, r),
                    10
                ),
                "scan_sub": self.create_subscription(
                    LaserScan,
                    f'/{robot}/scan',
                    lambda msg, r=robot: self.scan_callback(msg, r),
                    10
                ),
                "imu_sub": self.create_subscription(
                    Imu,
                    f'/{robot}/imu/data',
                    lambda msg, r=robot: self.imu_callback(msg, r),
                    10
                ),
            }
        self.robot_publishers = {}
        for robot in self.robot_names:
            self.robot_publishers[robot] = self.create_publisher(Twist, f'/{robot}/cmd_vel', 10)

        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Swarm Manager Node has been started.')

    def timer_callback(self):
        for robot, pub in self.robot_publishers.items():
            cmd = Twist()
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0
            pub.publish(cmd)
        self.get_logger().info("Sent forward commands to all robots")

    def odom_callback(self, msg, robot):
        self.robot_data[robot]["odom"] = msg
        self.get_logger().info(f'Received odometry from {robot}')

    def scan_callback(self, msg, robot):
        self.robot_data[robot]["scan"] = msg
        self.get_logger().info(f'Received laser scan from {robot}')

    def imu_callback(self, msg, robot):
        self.robot_data[robot]["imu"] = msg
        self.get_logger().info(f'Received IMU data from {robot}')

def main(args=None):
    rclpy.init(args=args)
    swarm_manager_node = SwarmManagerNode()
    rclpy.spin(swarm_manager_node)
    swarm_manager_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()