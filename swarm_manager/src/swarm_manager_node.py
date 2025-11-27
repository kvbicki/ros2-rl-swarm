#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from swarm_env import SwarmEnv
from rl_algorithms import PPOAgent
import numpy as np
import torch
import os
from datetime import datetime

class SwarmManagerNode(Node):
    def __init__(self):
        super().__init__('swarm_manager_node')

        # Parametry
        self.robot_names = ['robot_1', 'robot_2', 'robot_3', 'robot_4', 'robot_5', 'robot_6']
        self.training_mode = self.declare_parameter('training', True).value
        
        # Poprawiona ścieżka modelu - zapisuje w home directory
        default_path = os.path.join(os.path.expanduser('~'), 'swarm_models')
        self.model_path = self.declare_parameter('model_path', default_path).value
        os.makedirs(self.model_path, exist_ok=True)
        
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Training mode: {self.training_mode}')

        # Stan robotów
        self.robot_data = {
            r: {"odom": None, "scan": None, "imu": None} 
            for r in self.robot_names
        }

        # Subskrypcje do topików
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

        # Publishery do sterowania
        self.robot_publishers = {
            r: self.create_publisher(Twist, f'/{r}/cmd_vel', 10) 
            for r in self.robot_names
        }

        # Środowisko i agent
        self.env = SwarmEnv(self.robot_names)
        obs_dim = 15 * len(self.robot_names)
        action_dim = 2 * len(self.robot_names)
        self.agent = PPOAgent(obs_dim, action_dim, lr=3e-4, gamma=0.99)

        # Załaduj model jeśli istnieje
        self.load_model()

        # Timer sterowania (10 Hz)
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # Liczniki treningowe
        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.max_episode_steps = 1000
        self.total_steps = 0
        
        # Statystyki
        self.episode_rewards = []
        self.best_reward = float('-inf')

        self.get_logger().info('Swarm Manager Node started successfully!')

    def control_callback(self):
        """Główna pętla sterowania"""
        # Sprawdź czy mamy dane od wszystkich robotów
        if not all(self.robot_data[r]["odom"] is not None for r in self.robot_names):
            return

        # Aktualizuj środowisko danymi z ROS
        self.env.update_from_ros_data(self.robot_data)
        
        # Pobierz obserwacje
        obs = self.env._get_observations().flatten()
        
        # Sprawdź czy obserwacje są poprawne
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            self.get_logger().warn('Invalid observations detected, skipping step')
            return

        # Agent wybiera akcję
        action, logprob = self.agent.act(obs)
        action = action.reshape(len(self.robot_names), 2)

        # Publikuj komendy sterowania
        for i, r in enumerate(self.robot_names):
            cmd = Twist()
            cmd.linear.x = float(np.clip(action[i, 0], -1.0, 1.0))
            cmd.angular.z = float(np.clip(action[i, 1], -2.0, 2.0))
            self.robot_publishers[r].publish(cmd)

        # Trening
        if self.training_mode:
            # Oblicz nagrody
            rewards = self.env._calculate_rewards()
            total_reward = np.sum(rewards)
            
            # Sprawdź warunek zakończenia
            done = self.env._check_done()
            
            # Zapisz transition
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

            # Warunek zakończenia epizodu
            episode_done = done or self.step_count >= self.max_episode_steps
            
            if episode_done:
                # Trenuj agenta na zebranych danych
                if len(self.agent.memory) > 0:
                    self.get_logger().info(
                        f'Training on {len(self.agent.memory)} transitions...'
                    )
                    self.agent.train()
                
                # Loguj statystyki
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
                
                # Zapisz model jeśli jest najlepszy
                if self.episode_reward > self.best_reward:
                    self.best_reward = self.episode_reward
                    self.save_model(best=True)
                    self.get_logger().info(f'New best model saved! Reward: {self.best_reward:.2f}')
                
                # Zapisuj checkpoint co 10 epizodów
                if self.episode_count % 10 == 0:
                    self.save_model()
                
                # Zapisuj szczegółowe checkpointy co 50 epizodów
                if self.episode_count % 50 == 0:
                    self.save_checkpoint()
                
                # Reset epizodu
                self.reset_episode()

    def reset_episode(self):
        """Resetuj epizod (soft reset bez restartowania Gazebo)"""
        self.get_logger().info('Resetting episode...')
        
        # Zatrzymaj wszystkie roboty
        stop_cmd = Twist()
        for r in self.robot_names:
            self.robot_publishers[r].publish(stop_cmd)
        
        # Reset środowiska (tylko wewnętrzny stan)
        self.env.reset()
        
        # Reset liczników
        self.step_count = 0
        self.episode_count += 1
        self.episode_reward = 0.0
        
        # Krótka pauza aby roboty się zatrzymały
        import time
        time.sleep(0.5)

    def save_model(self, best=False):
        """Zapisz model"""
        try:
            if best:
                model_path = os.path.join(self.model_path, 'ppo_best.pth')
            else:
                model_path = os.path.join(self.model_path, f'ppo_ep{self.episode_count}.pth')
            
            torch.save(self.agent.policy.state_dict(), model_path)
            self.get_logger().info(f'Model saved to: {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save model: {e}')

    def save_checkpoint(self):
        """Zapisz pełny checkpoint z metadanymi"""
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
            
            # Zapisz też statystyki do CSV
            stats_path = os.path.join(self.model_path, 'training_stats.csv')
            with open(stats_path, 'a') as f:
                if self.episode_count == 50:  # Pierwszy zapis - dodaj header
                    f.write('episode,total_steps,reward,avg_reward_100\n')
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                f.write(f'{self.episode_count},{self.total_steps},{self.episode_reward:.2f},{avg_reward:.2f}\n')
                
        except Exception as e:
            self.get_logger().error(f'Failed to save checkpoint: {e}')

    def load_model(self):
        """Załaduj najlepszy model jeśli istnieje"""
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
        """Callback dla odometrii"""
        self.robot_data[robot]["odom"] = msg

    def scan_callback(self, msg, robot):
        """Callback dla lidaru"""
        self.robot_data[robot]["scan"] = msg

    def imu_callback(self, msg, robot):
        """Callback dla IMU"""
        self.robot_data[robot]["imu"] = msg


def main(args=None):
    rclpy.init(args=args)
    node = SwarmManagerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
        # Zapisz model przed wyjściem
        if node.training_mode:
            node.save_model()
            node.save_checkpoint()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()