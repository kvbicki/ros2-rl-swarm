#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from swarm_env import SwarmEnv
from rl_algorithms import PPOAgent
import numpy as np
import os
import subprocess
import time
import torch
import sys

class SwarmManagerNode(Node):
    def __init__(self):
        super().__init__('swarm_manager_node')
        sys.stdout.reconfigure(line_buffering=True)
        self.get_logger().info(">>> SWARM MANAGER INITIALIZATION <<<")

        self.robot_names = ['robot_1', 'robot_2', 'robot_3', 'robot_4', 'robot_5', 'robot_6']
        self.training_mode = self.declare_parameter('training', True).value
        
        default_path = os.path.join(os.path.expanduser('~'), 'swarm_models')
        self.model_path = self.declare_parameter('model_path', default_path).value
        os.makedirs(self.model_path, exist_ok=True)
        
        stats_path = os.path.join(self.model_path, 'training_stats.csv')
        if not os.path.exists(stats_path):
            with open(stats_path, 'w') as f:
                f.write('episode,steps,reward,avg_reward_100,collision_count\n')

        self.robot_data = {
            r: {"odom": None, "scan": None, "imu": None} 
            for r in self.robot_names
        }

        for r in self.robot_names:
            self.create_subscription(Odometry, f'/{r}/odom', lambda msg, r=r: self.odom_callback(msg, r), 10)
            self.create_subscription(LaserScan, f'/{r}/scan', lambda msg, r=r: self.scan_callback(msg, r), 10)
            self.create_subscription(Imu, f'/{r}/imu/data', lambda msg, r=r: self.imu_callback(msg, r), 10)

        self.robot_publishers = {
            r: self.create_publisher(Twist, f'/{r}/cmd_vel', 10) 
            for r in self.robot_names
        }

        self.env = SwarmEnv(self.robot_names)
        
        single_robot_obs_dim = 15
        single_robot_action_dim = 2
        
        self.agent = PPOAgent(
            single_robot_obs_dim, 
            single_robot_action_dim, 
            lr=3e-4, 
            gamma=0.99,
            entropy_coef=0.05
        )

        self.load_model()
        
        self.control_timer = self.create_timer(0.1, self.control_callback)

        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.max_episode_steps = 400
        self.total_steps = 0
        self.wait_counter = 0
        
        self.episode_rewards = []
        self.best_reward = float('-inf')
        
        self.get_logger().info(">>> MANAGER READY - WAITING FOR GAZEBO DATA <<<")

    def control_callback(self):
        missing_data = []
        for r in self.robot_names:
            if self.robot_data[r]["odom"] is None: missing_data.append(f"{r}:odom")

        if missing_data:
            self.wait_counter += 1
            if self.wait_counter % 20 == 0:
                self.get_logger().warn(f"WAITING FOR SENSOR DATA: {', '.join(missing_data)}")
            return

        self.wait_counter = 0

        self.env.update_from_ros_data(self.robot_data)
        obs_all = self.env._get_observations()
        
        if np.any(np.isnan(obs_all)) or np.any(np.isinf(obs_all)):
            self.get_logger().error("DETECTED NaN/Inf IN OBSERVATIONS! SKIPPING STEP.")
            return

        actions, logprobs = self.agent.act(obs_all)
        
        for i, r in enumerate(self.robot_names):
            cmd = Twist()
            cmd.linear.x = float(np.clip(actions[i, 0], -1.0, 1.0))
            cmd.angular.z = float(np.clip(actions[i, 1], -2.0, 2.0))
            self.robot_publishers[r].publish(cmd)

        if self.training_mode:
            rewards = self.env._calculate_rewards()
            total_step_reward = np.sum(rewards)
            done = self.env._check_done()
            
            for i in range(len(self.robot_names)):
                self.agent.store_transition(
                    obs_all[i], actions[i], logprobs[i], rewards[i], done
                )
            
            self.episode_reward += total_step_reward
            self.step_count += 1
            self.total_steps += 1

            if self.step_count % 50 == 0:
                self.get_logger().info(f"Step {self.step_count}/{self.max_episode_steps}, Reward: {self.episode_reward:.1f}")

            episode_done = done or self.step_count >= self.max_episode_steps
            
            if episode_done:
                self.handle_episode_end()
                self.reset_episode()

    def handle_episode_end(self):
        if len(self.agent.memory) > 0:
            self.get_logger().info("TRAINING NETWORK...")
            self.agent.train()
        
        self.episode_rewards.append(self.episode_reward)
        avg = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        
        self.get_logger().info(
            f"\n{'#'*40}\n"
            f"EPISODE {self.episode_count} ENDED\n"
            f"REWARD: {self.episode_reward:.2f}\n"
            f"AVG (100): {avg:.2f}\n"
            f"COLLISIONS: {self.env.collision_count}\n"
            f"{'#'*40}"
        )

        try:
            stats_path = os.path.join(self.model_path, 'training_stats.csv')
            with open(stats_path, 'a') as f:
                f.write(f'{self.episode_count},{self.step_count},{self.episode_reward:.2f},{avg:.2f},{self.env.collision_count}\n')
        except Exception as e:
            self.get_logger().error(f"CSV SAVE ERROR: {e}")

        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.save_model(best=True)
        
        if self.episode_count % 10 == 0:
            self.save_model()
            self.save_checkpoint()

    def reset_episode(self):
        self.get_logger().info("RESETTING POSITIONS...")
        stop_cmd = Twist()
        for r in self.robot_names:
            self.robot_publishers[r].publish(stop_cmd)
        
        time.sleep(0.2)
        
        positions = [
            (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
            (0.5, 0.25), (0.5, 0.75), (1.0, 0.5)
        ]
        
        for i, (x_pos, y_pos) in enumerate(positions):
            robot_name = f"robot_{i+1}"
            gz_cmd = [
                'gz', 'service', '-s', '/world/empty/set_pose',
                '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000', '--req',
                f'name: "{robot_name}" position: {{x: {x_pos}, y: {y_pos}, z: 0.5}} orientation: {{w: 1.0}}'
            ]
            try:
                subprocess.run(gz_cmd, capture_output=True, timeout=1)
            except:
                pass
        
        time.sleep(0.5)
        self.env.reset()
        self.step_count = 0
        self.episode_count += 1
        self.episode_reward = 0.0

    def save_model(self, best=False):
        try:
            name = 'ppo_best.pth' if best else f'ppo_ep{self.episode_count}.pth'
            path = os.path.join(self.model_path, name)
            torch.save(self.agent.policy.state_dict(), path)
        except Exception as e:
            self.get_logger().error(f"SAVE ERROR: {e}")

    def save_checkpoint(self):
        try:
            path = os.path.join(self.model_path, f'checkpoint_ep{self.episode_count}.pth')
            checkpoint = {
                'episode': self.episode_count,
                'model_state': self.agent.policy.state_dict(),
                'optimizer_state': self.agent.optimizer.state_dict(),
                'rewards': self.episode_rewards
            }
            torch.save(checkpoint, path)
        except Exception as e:
            self.get_logger().error(f"CHECKPOINT ERROR: {e}")

    def load_model(self):
        path = os.path.join(self.model_path, 'ppo_best.pth')
        if os.path.exists(path):
            self.agent.policy.load_state_dict(torch.load(path))
            self.get_logger().info("MODEL LOADED")

    def odom_callback(self, msg, robot): self.robot_data[robot]["odom"] = msg
    def scan_callback(self, msg, robot): self.robot_data[robot]["scan"] = msg
    def imu_callback(self, msg, robot): self.robot_data[robot]["imu"] = msg

def main(args=None):
    rclpy.init(args=args)
    node = SwarmManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.training_mode:
            node.save_model()
            node.save_checkpoint()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()