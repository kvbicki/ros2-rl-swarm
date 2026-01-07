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
from datetime import datetime
import subprocess
import time

class SwarmManagerNode(Node):
    def __init__(self):
        super().__init__('swarm_manager_node')

        self.robot_names = ['robot_1', 'robot_2', 'robot_3', 'robot_4', 'robot_5', 'robot_6']
        self.training_mode = self.declare_parameter('training', True).value
        
        default_path = '/ros2_ws/src/ros2-rl-swarm/swarm_manager/models'
        self.model_path = self.declare_parameter('model_path', default_path).value
        os.makedirs(self.model_path, exist_ok=True)
        
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Training mode: {self.training_mode}')

        self.robot_data = {
            r: {"odom": None, "scan": None, "imu": None} 
            for r in self.robot_names
        }

        for r in self.robot_names:
            self.create_subscription(
                Odometry, f'/{r}/odom', 
                lambda msg, r=r: self.odom_callback(msg, r), 10
            )
            self.create_subscription(
                LaserScan, f'/{r}/scan', 
                lambda msg, r=r: self.scan_callback(msg, r), 10
            )
            self.create_subscription(
                Imu, f'/{r}/imu/data', 
                lambda msg, r=r: self.imu_callback(msg, r), 10
            )

        self.robot_publishers = {
            r: self.create_publisher(Twist, f'/{r}/cmd_vel', 10) 
            for r in self.robot_names
        }

        self.env = SwarmEnv(self.robot_names)
        obs_dim = 15 * len(self.robot_names)
        action_dim = 2 * len(self.robot_names)
        self.agent = PPOAgent(obs_dim, action_dim, lr=3e-4, gamma=0.99)

        self.load_model()

        self.control_timer = self.create_timer(0.1, self.control_callback)

        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.max_episode_steps = 400
        self.total_steps = 0
        
        self.episode_rewards = []
        self.best_reward = float('-inf')

        self.get_logger().info('Swarm Manager Node started successfully!')

    def control_callback(self):
        if not all(self.robot_data[r]["odom"] is not None for r in self.robot_names):
            return

        self.env.update_from_ros_data(self.robot_data)
        
        obs = self.env._get_observations().flatten()
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -10, 10)
        
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            self.get_logger().warn('Invalid observations detected, skipping step')
            return

        action, logprob = self.agent.act(obs)
        action = np.asarray(action).flatten()
        if len(action) != len(self.robot_names) * 2:
            self.get_logger().warn(f'Action size mismatch: {len(action)}')
            action = np.zeros(len(self.robot_names) * 2)
        action = action.reshape(len(self.robot_names), 2)

        for i, r in enumerate(self.robot_names):
            cmd = Twist()
            cmd.linear.x = float(np.clip(action[i, 0], -1.0, 1.0))
            cmd.angular.z = float(np.clip(action[i, 1], -2.0, 2.0))
            self.robot_publishers[r].publish(cmd)

        if self.training_mode:
            rewards = self.env._calculate_rewards()
            total_reward = np.sum(rewards)
            
            done = self.env._check_done()
            
            self.agent.store_transition(
                obs, 
                action.flatten(), 
                logprob, 
                total_reward, 
                done
            )
            
            self.episode_reward += total_reward
            self.step_count += 1
            self.total_steps += 1

            episode_done = done or self.step_count >= self.max_episode_steps
            
            if episode_done:
                if len(self.agent.memory) > 0:
                    self.get_logger().info(
                        f'Training on {len(self.agent.memory)} transitions...'
                    )
                    self.agent.train()
                
                self.episode_rewards.append(self.episode_reward)
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                
                self.get_logger().info(
                    f'\n{"="*60}\n'
                    f'Episode {self.episode_count} finished:\n'
                    f'  Steps: {self.step_count}\n'
                    f'  Total reward: {self.episode_reward:.2f}\n'
                    f'  Avg reward (last 100): {avg_reward:.2f}\n'
                    f'  Total steps: {self.total_steps}\n'
                    f'  Success: {"Yes" if done else "No"}\n'
                    f'{"="*60}'
                )
                
                self.save_stats_to_csv()
                
                if self.episode_reward > self.best_reward:
                    self.best_reward = self.episode_reward
                    self.save_model(best=True)
                    self.get_logger().info(f'New best model saved! Reward: {self.best_reward:.2f}')
                
                if self.episode_count % 10 == 0:
                    self.save_model()
                
                if self.episode_count % 50 == 0:
                    self.save_checkpoint()
                
                self.reset_episode()

    def reset_episode(self):
        self.get_logger().info('Resetting episode - teleporting robots...')
        
        stop_cmd = Twist()
        for r in self.robot_names:
            self.robot_publishers[r].publish(stop_cmd)
        
        time.sleep(0.3)
        
        positions = [
            (0.0, 0.0),
            (0.0, 0.5),
            (0.0, 1.0),
            (0.5, 0.25),
            (0.5, 0.75),
            (1.0, 0.5)
        ]
        
        for i, (x_pos, y_pos) in enumerate(positions):
            robot_name = f"robot_{i+1}"
            
            gz_cmd = [
                'gz', 'service',
                '-s', '/world/empty/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req',
                f'name: "{robot_name}" '
                f'position: {{x: {x_pos}, y: {y_pos}, z: 0.5}} '
                f'orientation: {{w: 1.0, x: 0.0, y: 0.0, z: 0.0}}'
            ]
            
            try:
                subprocess.run(gz_cmd, capture_output=True, text=True, timeout=2)
            except Exception:
                pass
        
        time.sleep(0.5)
        
        self.env.reset()
        
        self.step_count = 0
        self.episode_count += 1
        self.episode_reward = 0.0
        
        self.get_logger().info('✓ Episode reset complete!')

    def save_model(self, best=False):
        try:
            if best:
                model_path = os.path.join(self.model_path, 'ppo_best.pth')
            else:
                model_path = os.path.join(self.model_path, f'ppo_ep{self.episode_count}.pth')
            
            torch.save(self.agent.policy.state_dict(), model_path)
            self.get_logger().info(f'Model saved to: {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save model: {e}')

    def save_stats_to_csv(self):
        stats_path = os.path.join(self.model_path, 'training_stats.csv')
        
        file_exists = os.path.isfile(stats_path) and os.path.getsize(stats_path) > 0
        
        env_stats = self.env.get_statistics()
        collision_count = env_stats.get('collision_count', 0)
        
        if len(self.episode_rewards) == 0:
            avg_reward = 0.0
        else:
            avg_reward = np.mean(self.episode_rewards[-100:])

        try:
            with open(stats_path, 'a') as f:
                if not file_exists:
                    f.write('episode,total_steps,reward,avg_reward_100,collision_count\n')
                
                f.write(f'{self.episode_count},{self.total_steps},{self.episode_reward:.2f},{avg_reward:.2f},{collision_count}\n')
        except Exception as e:
            self.get_logger().error(f'Failed to write to CSV: {e}')

    def save_checkpoint(self):
        try:
            checkpoint_path = os.path.join(
                self.model_path, 
                f'checkpoint_ep{self.episode_count}.pth'
            )
            
            checkpoint = {
                'episode': self.episode_count,
                'total_steps': self.total_steps,
                'model_state_dict': self.agent.policy.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'episode_rewards': self.episode_rewards,
                'best_reward': self.best_reward,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.get_logger().info(f'Checkpoint saved to: {checkpoint_path}')
                
        except Exception as e:
            self.get_logger().error(f'Failed to save checkpoint: {e}')

    def load_model(self):
        best_model_path = os.path.join(self.model_path, 'ppo_best.pth')
        
        if os.path.exists(best_model_path):
            try:
                self.agent.policy.load_state_dict(torch.load(best_model_path))
                self.get_logger().info(f'Loaded best model from: {best_model_path}')
            except Exception as e:
                self.get_logger().warn(f'Failed to load model: {e}')
        else:
            self.get_logger().info('No existing model found, starting from scratch')

    def odom_callback(self, msg, robot):
        self.robot_data[robot]["odom"] = msg

    def scan_callback(self, msg, robot):
        self.robot_data[robot]["scan"] = msg

    def imu_callback(self, msg, robot):
        self.robot_data[robot]["imu"] = msg


def main(args=None):
    rclpy.init(args=args)
    node = SwarmManagerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
        if node.training_mode:
            node.save_model()
            node.save_checkpoint()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()