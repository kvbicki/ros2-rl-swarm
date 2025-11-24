#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from swarm_env import SwarmEnv
from rl_algorithms import PPOAgent
import numpy as np
import torch
import os

class SwarmManagerNode(Node):
    def __init__(self):
        super().__init__('swarm_manager_node')

        self.robot_names = ['robot_1', 'robot_2', 'robot_3', 'robot_4', 'robot_5', 'robot_6']
        self.robot_data = {r: {"odom": None, "scan": None, "imu": None} for r in self.robot_names}

        self.training_mode = self.declare_parameter('training', True).value
        self.model_path = self.declare_parameter('model_path', 'models/').value

        for r in self.robot_names:
            self.create_subscription(Odometry, f'/{r}/odom', lambda msg, r=r: self.odom_callback(msg, r), 10)
            self.create_subscription(LaserScan, f'/{r}/scan', lambda msg, r=r: self.scan_callback(msg, r), 10)
            self.create_subscription(Imu, f'/{r}/imu/data', lambda msg, r=r: self.imu_callback(msg, r), 10)

        self.robot_publishers = {r: self.create_publisher(Twist, f'/{r}/cmd_vel', 10) for r in self.robot_names}
        self.env = SwarmEnv(self.robot_names)

        input_dim = 15
        output_dim = 2
        self.agent = PPOAgent(input_dim * len(self.robot_names), output_dim * len(self.robot_names))

        self.load_model()

        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.save_timer = self.create_timer(60.0, self.save_model)

        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0

        self.get_logger().info('Swarm Manager started (PPO only)')

    def control_callback(self):
        if not all(self.robot_data[r]["odom"] is not None for r in self.robot_names):
            return

        self.env.update_from_ros_data(self.robot_data)
        observations = self.env._get_observations()
        flat_obs = observations.flatten()

        actions, logprobs = self.agent.act(flat_obs)
        actions = actions.reshape(len(self.robot_names), 2)

        for i, robot_name in enumerate(self.robot_names):
            cmd = Twist()
            cmd.linear.x = float(np.clip(actions[i, 0], -1.0, 1.0))
            cmd.angular.z = float(np.clip(actions[i, 1], -2.0, 2.0))
            self.robot_publishers[robot_name].publish(cmd)

        if self.training_mode:
            reward_sum = np.sum(self.env._calculate_rewards())
            self.agent.store_transition(flat_obs, actions.flatten(), logprobs, reward_sum, False)
            self.episode_reward += reward_sum
            if self.step_count % 2048 == 0:
                self.agent.train()

        self.step_count += 1
        if self.step_count % 1000 == 0:
            self.get_logger().info(f'Episode {self.episode_count}: Reward = {self.episode_reward:.2f}')
            self.episode_count += 1
            self.episode_reward = 0
            self.env.reset()

    def save_model(self):
        if not self.training_mode:
            return
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.agent.policy.state_dict(), f'{self.model_path}/ppo_model.pth')
        self.get_logger().info('Model saved')

    def load_model(self):
        model_file = f'{self.model_path}/ppo_model.pth'
        if os.path.exists(model_file):
            self.agent.policy.load_state_dict(torch.load(model_file))
            self.get_logger().info('Loaded PPO model')

    def odom_callback(self, msg, robot):
        self.robot_data[robot]["odom"] = msg

    def scan_callback(self, msg, robot):
        self.robot_data[robot]["scan"] = msg

    def imu_callback(self, msg, robot):
        self.robot_data[robot]["imu"] = msg

def main(args=None):
    rclpy.init(args=args)
    node = SwarmManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
