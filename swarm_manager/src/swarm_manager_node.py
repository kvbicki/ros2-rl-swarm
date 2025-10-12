#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from swarm_env import SwarmEnv
from rl_algorithms import PPOAgent, SACAgent, DQNAgent
import numpy as np
import torch
import pickle
import os

class SwarmManagerNode(Node):
    def __init__(self):
        super().__init__('swarm_manager_node')
        
        self.robot_names = ['robot_1', 'robot_2', 'robot_3', 'robot_4', 'robot_5', 'robot_6']
        self.robot_data = {r: {"odom": None, "scan": None, "imu": None} for r in self.robot_names}
        
        self.algorithm = self.declare_parameter('algorithm', 'ppo').value
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
        
        if self.algorithm == 'ppo':
            self.agent = PPOAgent(input_dim * len(self.robot_names), output_dim * len(self.robot_names))
        elif self.algorithm == 'sac':
            self.agent = SACAgent(input_dim * len(self.robot_names), output_dim * len(self.robot_names))
        elif self.algorithm == 'dqn':
            self.agent = DQNAgent(input_dim * len(self.robot_names), output_dim * len(self.robot_names))
        
        self.load_model()
        
        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.save_timer = self.create_timer(60.0, self.save_model)
        
        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0
        
        self.get_logger().info(f'Swarm Manager started with {self.algorithm} algorithm')
    
    def control_callback(self):
        if not all(self.robot_data[r]["odom"] is not None for r in self.robot_names):
            return
        
        self.env.update_from_ros_data(self.robot_data)
        observations = self.env._get_observations()
        flat_obs = observations.flatten()
        
        if self.algorithm == 'ppo':
            actions, logprobs = self.agent.act(flat_obs)
            actions = actions.reshape(len(self.robot_names), 2)
        elif self.algorithm == 'sac':
            actions = self.agent.act(flat_obs, evaluate=not self.training_mode)
            actions = actions.reshape(len(self.robot_names), 2)
        elif self.algorithm == 'dqn':
            action_idx = self.agent.act(flat_obs)
            actions = self.decode_discrete_actions(action_idx)
        
        next_obs, rewards, done, info = self.env.step(actions)
        
        for i, robot_name in enumerate(self.robot_names):
            cmd = Twist()
            cmd.linear.x = float(np.clip(actions[i, 0], -1.0, 1.0))
            cmd.angular.z = float(np.clip(actions[i, 1], -2.0, 2.0))
            self.robot_publishers[robot_name].publish(cmd)
        
        if self.training_mode:
            reward_sum = np.sum(rewards)
            self.episode_reward += reward_sum
            
            if self.algorithm == 'ppo':
                self.agent.store_transition(flat_obs, actions.flatten(), logprobs, reward_sum, done)
                if self.step_count % 2048 == 0:
                    self.agent.train()
            elif self.algorithm == 'sac':
                self.agent.remember(flat_obs, actions.flatten(), reward_sum, next_obs.flatten(), done)
                self.agent.train()
            elif self.algorithm == 'dqn':
                self.agent.remember(flat_obs, action_idx, reward_sum, next_obs.flatten(), done)
                self.agent.replay()
                if self.step_count % 100 == 0:
                    self.agent.update_target_network()
        
        self.step_count += 1
        
        if done or self.step_count % 1000 == 0:
            self.get_logger().info(f'Episode {self.episode_count}: Reward = {self.episode_reward:.2f}')
            self.episode_count += 1
            self.episode_reward = 0
            self.env.reset()
    
    def decode_discrete_actions(self, action_idx):
        num_actions_per_robot = 9
        actions = np.zeros((len(self.robot_names), 2))
        
        for i in range(len(self.robot_names)):
            robot_action = (action_idx // (num_actions_per_robot ** i)) % num_actions_per_robot
            linear_vel = (robot_action // 3 - 1) * 0.5
            angular_vel = (robot_action % 3 - 1) * 1.0
            actions[i] = [linear_vel, angular_vel]
        
        return actions
    
    def save_model(self):
        if not self.training_mode:
            return
        
        os.makedirs(self.model_path, exist_ok=True)
        
        if self.algorithm == 'ppo':
            torch.save(self.agent.policy.state_dict(), f'{self.model_path}/ppo_model.pth')
        elif self.algorithm == 'sac':
            torch.save({
                'actor': self.agent.actor.state_dict(),
                'critic1': self.agent.critic1.state_dict(),
                'critic2': self.agent.critic2.state_dict()
            }, f'{self.model_path}/sac_model.pth')
        elif self.algorithm == 'dqn':
            torch.save(self.agent.q_network.state_dict(), f'{self.model_path}/dqn_model.pth')
        
        self.get_logger().info('Model saved')
    
    def load_model(self):
        if self.algorithm == 'ppo' and os.path.exists(f'{self.model_path}/ppo_model.pth'):
            self.agent.policy.load_state_dict(torch.load(f'{self.model_path}/ppo_model.pth'))
        elif self.algorithm == 'sac' and os.path.exists(f'{self.model_path}/sac_model.pth'):
            checkpoint = torch.load(f'{self.model_path}/sac_model.pth')
            self.agent.actor.load_state_dict(checkpoint['actor'])
            self.agent.critic1.load_state_dict(checkpoint['critic1'])
            self.agent.critic2.load_state_dict(checkpoint['critic2'])
        elif self.algorithm == 'dqn' and os.path.exists(f'{self.model_path}/dqn_model.pth'):
            self.agent.q_network.load_state_dict(torch.load(f'{self.model_path}/dqn_model.pth'))
    
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