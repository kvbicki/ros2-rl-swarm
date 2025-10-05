#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from rl_agent import PolicyNet, save_policy, load_policy
from swarm_env import SwarmEnv
import torch

class SwarmManagerNode(Node):
    def __init__(self):
        super().__init__('swarm_manager_node')
        self.robot_names = ['robot_1','robot_2','robot_3','robot_4','robot_5','robot_6']
        self.robot_data = {r: {"odom": None, "scan": None, "imu": None} for r in self.robot_names}
        self.robot_subscriptions = {}
        for r in self.robot_names:
            self.robot_subscriptions[r] = {
                "odom_sub": self.create_subscription(Odometry,f'/{r}/odom',lambda msg, r=r: self.odom_callback(msg,r),10),
                "scan_sub": self.create_subscription(LaserScan,f'/{r}/scan',lambda msg, r=r: self.scan_callback(msg,r),10),
                "imu_sub": self.create_subscription(Imu,f'/{r}/imu/data',lambda msg, r=r: self.imu_callback(msg,r),10)
            }
        self.robot_publishers = {r:self.create_publisher(Twist,f'/{r}/cmd_vel',10) for r in self.robot_names}
        self.env = SwarmEnv(self.robot_names)
        self.policy = PolicyNet(input_dim=3, output_dim=2)
        load_policy(self.policy)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Swarm Manager Node has been started.')

    def timer_callback(self):
        obs = torch.tensor(self.env.reset(), dtype=torch.float32)
        actions = self.policy(obs).detach().numpy()
        for i, robot in enumerate(self.robot_names):
            cmd = Twist()
            cmd.linear.x = actions[i,0]
            cmd.angular.z = actions[i,1]
            self.robot_publishers[robot].publish(cmd)
        save_policy(self.policy)

    def odom_callback(self,msg,robot):
        self.robot_data[robot]["odom"]=msg

    def scan_callback(self,msg,robot):
        self.robot_data[robot]["scan"]=msg

    def imu_callback(self,msg,robot):
        self.robot_data[robot]["imu"]=msg

def main(args=None):
    rclpy.init(args=args)
    node = SwarmManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()