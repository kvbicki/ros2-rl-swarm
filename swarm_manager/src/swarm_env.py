#!/usr/bin/env python3
import numpy as np
from typing import Dict, Tuple, Optional
import math

class SwarmEnv:
    def __init__(self, robot_names: list, observation_space_dim: int = 15, action_space_dim: int = 2):
        """
        Środowisko dla roju robotów.
        
        Args:
            robot_names: Lista nazw robotów
            observation_space_dim: Wymiar przestrzeni obserwacji dla każdego robota
            action_space_dim: Wymiar przestrzeni akcji (linear_vel, angular_vel)
        """
        self.robot_names = robot_names
        self.num_robots = len(robot_names)
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        
        # Parametry środowiska
        self.max_linear_vel = 1.0  # m/s
        self.max_angular_vel = 2.0  # rad/s
        self.collision_threshold = 0.5  # m
        self.target_position = np.array([5.0, 5.0])  # Cel dla formacji
        
        # Stan robotów
        self.robot_positions = {}
        self.robot_velocities = {}
        self.robot_orientations = {}
        self.previous_distances = {}
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset środowiska do stanu początkowego."""
        # Inicjalizacja pozycji robotów w formacji koła
        for i, robot_name in enumerate(self.robot_names):
            angle = 2 * math.pi * i / self.num_robots
            radius = 2.0
            self.robot_positions[robot_name] = np.array([
                radius * math.cos(angle),
                radius * math.sin(angle)
            ])
            self.robot_velocities[robot_name] = np.zeros(2)
            self.robot_orientations[robot_name] = angle
            self.previous_distances[robot_name] = np.linalg.norm(
                self.robot_positions[robot_name] - self.target_position
            )
        
        return self._get_observations()
    
    def _get_observations(self) -> np.ndarray:
        """
        Generuje obserwacje dla wszystkich robotów.
        Każda obserwacja zawiera:
        - Względną pozycję do celu (2)
        - Prędkość własną (2)
        - Orientację (1)
        - Najbliższe przeszkody z lidaru (5)
        - Pozycje najbliższych sąsiadów (5)
        """
        observations = np.zeros((self.num_robots, self.observation_space_dim))
        
        for i, robot_name in enumerate(self.robot_names):
            obs = []
            
            # Względna pozycja do celu
            rel_target = self.target_position - self.robot_positions[robot_name]
            obs.extend(rel_target / 10.0)  # Normalizacja
            
            # Prędkość własna
            obs.extend(self.robot_velocities[robot_name])
            
            # Orientacja (sin, cos)
            obs.append(math.sin(self.robot_orientations[robot_name]))
            obs.append(math.cos(self.robot_orientations[robot_name]))
            
            # Symulacja danych z lidaru (5 promieni)
            lidar_ranges = self._simulate_lidar(robot_name, num_rays=5)
            obs.extend(lidar_ranges)
            
            # Pozycje najbliższych sąsiadów
            neighbor_info = self._get_neighbor_info(robot_name, num_neighbors=2)
            obs.extend(neighbor_info)
            
            observations[i, :len(obs)] = obs
        
        return observations
    
    def _simulate_lidar(self, robot_name: str, num_rays: int = 5) -> list:
        """Symuluje dane z lidaru."""
        max_range = 10.0
        lidar_data = []
        
        for i in range(num_rays):
            angle = self.robot_orientations[robot_name] + (i - num_rays//2) * (math.pi / 4)
            
            # Sprawdzanie kolizji z innymi robotami
            min_distance = max_range
            for other_name in self.robot_names:
                if other_name != robot_name:
                    rel_pos = self.robot_positions[other_name] - self.robot_positions[robot_name]
                    distance = np.linalg.norm(rel_pos)
                    
                    # Sprawdzenie czy robot jest w kierunku promienia
                    ray_dir = np.array([math.cos(angle), math.sin(angle)])
                    if np.dot(rel_pos, ray_dir) > 0:
                        min_distance = min(min_distance, distance)
            
            lidar_data.append(min_distance / max_range)  # Normalizacja
        
        return lidar_data
    
    def _get_neighbor_info(self, robot_name: str, num_neighbors: int = 2) -> list:
        """Zwraca informacje o najbliższych sąsiadach."""
        neighbor_info = []
        distances = []
        
        for other_name in self.robot_names:
            if other_name != robot_name:
                rel_pos = self.robot_positions[other_name] - self.robot_positions[robot_name]
                distances.append((np.linalg.norm(rel_pos), rel_pos))
        
        # Sortowanie po odległości
        distances.sort(key=lambda x: x[0])
        
        for i in range(min(num_neighbors, len(distances))):
            if i < len(distances):
                neighbor_info.extend(distances[i][1] / 10.0)  # Normalizacja
            else:
                neighbor_info.extend([0.0, 0.0])
        
        return neighbor_info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        """
        Wykonuje krok w środowisku.
        
        Args:
            actions: Tablica akcji [num_robots, 2] (linear_vel, angular_vel)
        
        Returns:
            observations: Nowe obserwacje
            rewards: Nagrody dla każdego robota
            done: Czy epizod się zakończył
            info: Dodatkowe informacje
        """
        dt = 0.1  # Krok czasowy
        
        # Aktualizacja stanów robotów
        for i, robot_name in enumerate(self.robot_names):
            # Ograniczenie akcji
            linear_vel = np.clip(actions[i, 0], -self.max_linear_vel, self.max_linear_vel)
            angular_vel = np.clip(actions[i, 1], -self.max_angular_vel, self.max_angular_vel)
            
            # Aktualizacja orientacji
            self.robot_orientations[robot_name] += angular_vel * dt
            
            # Aktualizacja prędkości i pozycji
            self.robot_velocities[robot_name] = np.array([
                linear_vel * math.cos(self.robot_orientations[robot_name]),
                linear_vel * math.sin(self.robot_orientations[robot_name])
            ])
            self.robot_positions[robot_name] += self.robot_velocities[robot_name] * dt
        
        # Obliczanie nagród
        rewards = self._calculate_rewards()
        
        # Sprawdzenie warunków zakończenia
        done = self._check_done()
        
        # Dodatkowe informacje
        info = self._get_info()
        
        observations = self._get_observations()
        
        return observations, rewards, done, info
    
    def _calculate_rewards(self) -> np.ndarray:
        """Oblicza nagrody dla każdego robota."""
        rewards = np.zeros(self.num_robots)
        
        for i, robot_name in enumerate(self.robot_names):
            reward = 0.0
            
            # Nagroda za zbliżanie się do celu
            current_distance = np.linalg.norm(self.robot_positions[robot_name] - self.target_position)
            distance_reward = self.previous_distances[robot_name] - current_distance
            reward += distance_reward * 10.0
            self.previous_distances[robot_name] = current_distance
            
            # Nagroda za utrzymywanie formacji
            formation_reward = 0.0
            for other_name in self.robot_names:
                if other_name != robot_name:
                    dist = np.linalg.norm(self.robot_positions[robot_name] - self.robot_positions[other_name])
                    # Idealna odległość to około 1.5m
                    formation_reward -= abs(dist - 1.5) * 0.1
            reward += formation_reward
            
            # Kara za kolizje
            for other_name in self.robot_names:
                if other_name != robot_name:
                    dist = np.linalg.norm(self.robot_positions[robot_name] - self.robot_positions[other_name])
                    if dist < self.collision_threshold:
                        reward -= 10.0
            
            # Nagroda za osiągnięcie celu
            if current_distance < 0.5:
                reward += 100.0
            
            rewards[i] = reward
        
        return rewards
    
    def _check_done(self) -> bool:
        """Sprawdza czy epizod się zakończył."""
        # Sprawdzenie czy wszystkie roboty osiągnęły cel
        all_at_target = True
        for robot_name in self.robot_names:
            distance = np.linalg.norm(self.robot_positions[robot_name] - self.target_position)
            if distance > 0.5:
                all_at_target = False
                break
        
        return all_at_target
    
    def _get_info(self) -> dict:
        """Zwraca dodatkowe informacje o stanie środowiska."""
        info = {
            "robot_positions": self.robot_positions.copy(),
            "robot_velocities": self.robot_velocities.copy(),
            "robot_orientations": self.robot_orientations.copy(),
            "target_position": self.target_position.copy(),
            "collisions": self._detect_collisions()
        }
        return info
    
    def _detect_collisions(self) -> list:
        """Wykrywa kolizje między robotami."""
        collisions = []
        for i, robot1 in enumerate(self.robot_names):
            for robot2 in self.robot_names[i+1:]:
                dist = np.linalg.norm(self.robot_positions[robot1] - self.robot_positions[robot2])
                if dist < self.collision_threshold:
                    collisions.append((robot1, robot2))
        return collisions
    
    def update_from_ros_data(self, robot_data: Dict) -> None:
        """
        Aktualizuje stan środowiska na podstawie danych z ROS.
        
        Args:
            robot_data: Słownik z danymi sensorycznymi z ROS
        """
        for robot_name in self.robot_names:
            if robot_name in robot_data:
                data = robot_data[robot_name]
                
                # Aktualizacja pozycji z odometrii
                if data["odom"] is not None:
                    self.robot_positions[robot_name] = np.array([
                        data["odom"].pose.pose.position.x,
                        data["odom"].pose.pose.position.y
                    ])
                    
                    # Konwersja kwaterniona na kąt Eulera (yaw)
                    q = data["odom"].pose.pose.orientation
                    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                    self.robot_orientations[robot_name] = yaw
                    
                    # Aktualizacja prędkości
                    self.robot_velocities[robot_name] = np.array([
                        data["odom"].twist.twist.linear.x,
                        data["odom"].twist.twist.linear.y
                    ])