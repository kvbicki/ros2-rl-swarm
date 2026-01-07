#!/usr/bin/env python3
import numpy as np
import math

class SwarmEnv:
    def __init__(self, robot_names: list, observation_space_dim: int = 15, action_space_dim: int = 2):
        self.robot_names = robot_names
        self.num_robots = len(robot_names)
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        
        self.max_linear_vel = 1.0
        self.max_angular_vel = 2.0
        self.collision_threshold = 0.35
        self.target_position = np.array([5.0, 5.0])
        self.ideal_neighbor_distance = 1.5
        
        self.robot_positions = {}
        self.robot_velocities = {}
        self.robot_orientations = {}
        self.robot_lidar_data = {}
        self.previous_distances = {}
        
        self.collision_count = 0
        self.success_count = 0
        self.reset()
    
    def reset(self):
        for i, robot_name in enumerate(self.robot_names):
            angle = 2 * math.pi * i / self.num_robots
            radius = 2.0
            self.robot_positions[robot_name] = np.array([
                radius * math.cos(angle),
                radius * math.sin(angle)
            ])
            self.robot_velocities[robot_name] = np.zeros(2)
            self.robot_orientations[robot_name] = angle
            self.robot_lidar_data[robot_name] = None
            self.previous_distances[robot_name] = np.linalg.norm(
                self.robot_positions[robot_name] - self.target_position
            )
        self.collision_count = 0
        return self._get_observations()
    
    def _get_observations(self):
        observations = np.zeros((self.num_robots, self.observation_space_dim))
        for i, robot_name in enumerate(self.robot_names):
            obs = []
            rel_target = self.target_position - self.robot_positions[robot_name]
            obs.extend(rel_target / 10.0)
            obs.extend(self.robot_velocities[robot_name] / self.max_linear_vel)
            obs.append(math.sin(self.robot_orientations[robot_name]))
            obs.append(math.cos(self.robot_orientations[robot_name]))
            
            if self.robot_lidar_data[robot_name] is not None:
                lidar = self._process_real_lidar(robot_name)
            else:
                lidar = self._simulate_lidar(robot_name)
            obs.extend(lidar)
            
            neighbors = self._get_neighbor_info(robot_name)
            obs.extend(neighbors)
            observations[i] = obs
        return observations
    
    def _calculate_rewards(self):
        rewards = np.zeros(self.num_robots)
        
        for i, robot_name in enumerate(self.robot_names):
            reward = 0.0
            
            current_dist = np.linalg.norm(self.robot_positions[robot_name] - self.target_position)
            diff = self.previous_distances[robot_name] - current_dist
            reward += np.clip(diff, -0.5, 0.5) * 5.0
            self.previous_distances[robot_name] = current_dist
            
            neighbor_count = 0
            formation_loss = 0.0
            for other in self.robot_names:
                if other != robot_name:
                    d = np.linalg.norm(self.robot_positions[robot_name] - self.robot_positions[other])
                    formation_loss += abs(d - self.ideal_neighbor_distance)
                    neighbor_count += 1
            if neighbor_count > 0:
                reward -= (formation_loss / neighbor_count) * 0.05
            
            for other in self.robot_names:
                if other != robot_name:
                    d = np.linalg.norm(self.robot_positions[robot_name] - self.robot_positions[other])
                    if d < self.collision_threshold:
                        reward -= 10.0
                        self.collision_count += 1
            
            if current_dist < 0.5:
                reward += 20.0
            
            speed = np.linalg.norm(self.robot_velocities[robot_name])
            if speed < 0.05:
                reward -= 0.2
            else:
                reward += 0.1
                
            rewards[i] = reward
            
        return rewards
    
    def _process_real_lidar(self, robot_name, num_samples=5):
        lidar_msg = self.robot_lidar_data[robot_name]
        ranges = np.array(lidar_msg.ranges)
        max_range = lidar_msg.range_max
        ranges[np.isinf(ranges)] = max_range
        ranges[np.isnan(ranges)] = max_range
        indices = np.linspace(0, len(ranges) - 1, num_samples, dtype=int)
        return (ranges[indices] / max_range).tolist()

    def _simulate_lidar(self, robot_name, num_rays=5):
        return [1.0] * num_rays

    def _get_neighbor_info(self, robot_name, num_neighbors=2):
        neighbor_info = []
        distances = []
        for other in self.robot_names:
            if other != robot_name:
                dist = np.linalg.norm(self.robot_positions[other] - self.robot_positions[robot_name])
                distances.append((dist, self.robot_positions[other] - self.robot_positions[robot_name]))
        distances.sort(key=lambda x: x[0])
        for i in range(num_neighbors):
            if i < len(distances):
                neighbor_info.extend(distances[i][1] / 10.0)
            else:
                neighbor_info.extend([0.0, 0.0])
        return neighbor_info

    def _check_done(self):
        all_at_target = True
        for robot_name in self.robot_names:
            if np.linalg.norm(self.robot_positions[robot_name] - self.target_position) > 0.5:
                all_at_target = False; break
        if all_at_target: self.success_count += 1
        return all_at_target

    def update_from_ros_data(self, robot_data):
        for r in self.robot_names:
            if r in robot_data and robot_data[r]["odom"]:
                odom = robot_data[r]["odom"]
                self.robot_positions[r] = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
                q = odom.pose.pose.orientation
                self.robot_orientations[r] = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y**2 + q.z**2))
                self.robot_velocities[r] = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y])
            if r in robot_data and robot_data[r]["scan"]:
                self.robot_lidar_data[r] = robot_data[r]["scan"]

    def get_statistics(self):
        return {
            "collision_count": self.collision_count,
            "success_count": self.success_count,
            "average_distance_to_target": np.mean([
                np.linalg.norm(pos - self.target_position)
                for pos in self.robot_positions.values()
            ])
        }