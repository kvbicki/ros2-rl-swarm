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
        self.collision_threshold = 0.4  # m - zmniejszone dla bezpieczeństwa
        self.target_position = np.array([5.0, 5.0])  # Cel dla formacji
        self.ideal_neighbor_distance = 1.5  # Idealna odległość między robotami
        
        # Stan robotów
        self.robot_positions = {}
        self.robot_velocities = {}
        self.robot_orientations = {}
        self.robot_lidar_data = {}  # Dodane dla rzeczywistych danych z lidaru
        self.previous_distances = {}
        
        # Statystyki
        self.collision_count = 0
        self.success_count = 0
        
        self.reset()
        self._validate_observations()

    def _validate_observations(self):
        """Sprawdź wymiary obserwacji."""
        obs = self._get_observations()
        assert obs.shape == (self.num_robots, self.observation_space_dim), \
            f"Expected {(self.num_robots, self.observation_space_dim)}, got {obs.shape}"
        return True

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
            self.robot_lidar_data[robot_name] = None
            self.previous_distances[robot_name] = np.linalg.norm(
                self.robot_positions[robot_name] - self.target_position
            )
        
        self.collision_count = 0
        
        return self._get_observations()
    
    def _get_observations(self) -> np.ndarray:
        """
        Generuje obserwacje dla wszystkich robotów.
        Każda obserwacja zawiera dokładnie 15 elementów:
        - Względną pozycję do celu (2)
        - Prędkość własną (2)
        - Orientację jako sin/cos (2)
        - Najbliższe przeszkody z lidaru (5)
        - Pozycje najbliższych sąsiadów (4) - 2 sąsiadów × 2 współrzędne
        """
        observations = np.zeros((self.num_robots, self.observation_space_dim))
        
        for i, robot_name in enumerate(self.robot_names):
            obs = []
            
            # 1. Względna pozycja do celu (2 elementy)
            rel_target = self.target_position - self.robot_positions[robot_name]
            rel_target_norm = rel_target / 10.0  # Normalizacja
            obs.extend(rel_target_norm)
            
            # 2. Prędkość własna (2 elementy)
            vel_norm = self.robot_velocities[robot_name] / self.max_linear_vel
            obs.extend(vel_norm)
            
            # 3. Orientacja jako sin i cos (2 elementy)
            obs.append(math.sin(self.robot_orientations[robot_name]))
            obs.append(math.cos(self.robot_orientations[robot_name]))
            
            # 4. Dane z lidaru (5 elementów)
            if self.robot_lidar_data[robot_name] is not None:
                lidar_ranges = self._process_real_lidar(robot_name, num_samples=5)
            else:
                lidar_ranges = self._simulate_lidar(robot_name, num_rays=5)
            obs.extend(lidar_ranges)
            
            # 5. Pozycje najbliższych sąsiadów (4 elementy = 2 sąsiadów × 2)
            neighbor_info = self._get_neighbor_info(robot_name, num_neighbors=2)
            obs.extend(neighbor_info)
            
            # Upewnij się że mamy dokładnie observation_space_dim elementów
            assert len(obs) == self.observation_space_dim, \
                f"Observation size mismatch: {len(obs)} != {self.observation_space_dim}"
            
            observations[i] = obs
        
        return observations
    
    def _process_real_lidar(self, robot_name: str, num_samples: int = 5) -> list:
        """
        Przetwarza rzeczywiste dane z lidaru.
        Próbkuje równomiernie z pełnego zakresu 360 stopni.
        """
        lidar_msg = self.robot_lidar_data[robot_name]
        ranges = np.array(lidar_msg.ranges)
        
        # Zamień inf na max_range
        max_range = lidar_msg.range_max
        ranges[np.isinf(ranges)] = max_range
        ranges[np.isnan(ranges)] = max_range
        
        # Równomierne próbkowanie
        total_points = len(ranges)
        indices = np.linspace(0, total_points - 1, num_samples, dtype=int)
        sampled_ranges = ranges[indices]
        
        # Normalizacja do [0, 1]
        normalized_ranges = sampled_ranges / max_range
        
        return normalized_ranges.tolist()
    
    def _simulate_lidar(self, robot_name: str, num_rays: int = 5) -> list:
        """
        Symuluje dane z lidaru gdy nie ma prawdziwych danych.
        """
        max_range = 10.0
        lidar_data = []
        
        for i in range(num_rays):
            # Rozłóż promienie równomiernie wokół robota
            angle_offset = (i - num_rays // 2) * (2 * math.pi / num_rays)
            angle = self.robot_orientations[robot_name] + angle_offset
            
            # Sprawdzanie kolizji z innymi robotami
            min_distance = max_range
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            
            for other_name in self.robot_names:
                if other_name != robot_name:
                    rel_pos = self.robot_positions[other_name] - self.robot_positions[robot_name]
                    distance = np.linalg.norm(rel_pos)
                    
                    # Sprawdź czy robot jest w kierunku promienia (±45 stopni)
                    if distance > 0:
                        rel_pos_norm = rel_pos / distance
                        dot_product = np.dot(rel_pos_norm, ray_dir)
                        if dot_product > 0.7:  # cos(45°) ≈ 0.7
                            min_distance = min(min_distance, distance)
            
            lidar_data.append(min_distance / max_range)  # Normalizacja
        
        return lidar_data
    
    def _get_neighbor_info(self, robot_name: str, num_neighbors: int = 2) -> list:
        """
        Zwraca względne pozycje najbliższych sąsiadów.
        Zawsze zwraca num_neighbors * 2 wartości (wypełnia zerami jeśli brak sąsiadów).
        """
        neighbor_info = []
        distances = []
        
        # Zbierz wszystkie odległości
        for other_name in self.robot_names:
            if other_name != robot_name:
                rel_pos = self.robot_positions[other_name] - self.robot_positions[robot_name]
                distance = np.linalg.norm(rel_pos)
                distances.append((distance, rel_pos))
        
        # Sortowanie po odległości
        distances.sort(key=lambda x: x[0])
        
        # Dodaj num_neighbors najbliższych
        for i in range(num_neighbors):
            if i < len(distances):
                # Normalizacja względnej pozycji
                rel_pos_norm = distances[i][1] / 10.0
                neighbor_info.extend(rel_pos_norm)
            else:
                # Wypełnij zerami jeśli brak sąsiadów
                neighbor_info.extend([0.0, 0.0])
        
        return neighbor_info
    
    def _calculate_rewards(self) -> np.ndarray:
        """
        Oblicza nagrody dla każdego robota.
        Składowe nagrody:
        1. Progress do celu
        2. Utrzymywanie formacji
        3. Kara za kolizje
        4. Bonus za osiągnięcie celu
        5. Kara za zbyt szybką jazdę (stabilność)
        """
        rewards = np.zeros(self.num_robots)
        
        for i, robot_name in enumerate(self.robot_names):
            reward = 0.0
            
            # 1. Nagroda za progress do celu
            current_distance = np.linalg.norm(
                self.robot_positions[robot_name] - self.target_position
            )
            distance_reward = (self.previous_distances[robot_name] - current_distance) * 10.0
            reward += distance_reward
            self.previous_distances[robot_name] = current_distance
            
            # 2. Nagroda za utrzymywanie idealnej formacji
            formation_reward = 0.0
            neighbor_count = 0
            for other_name in self.robot_names:
                if other_name != robot_name:
                    dist = np.linalg.norm(
                        self.robot_positions[robot_name] - self.robot_positions[other_name]
                    )
                    # Nagroda jest największa gdy dist ≈ ideal_neighbor_distance
                    distance_error = abs(dist - self.ideal_neighbor_distance)
                    formation_reward -= distance_error * 0.2
                    neighbor_count += 1
            
            if neighbor_count > 0:
                formation_reward /= neighbor_count
            reward += formation_reward
            
            # 3. Kara za kolizje
            collision_penalty = 0.0
            for other_name in self.robot_names:
                if other_name != robot_name:
                    dist = np.linalg.norm(
                        self.robot_positions[robot_name] - self.robot_positions[other_name]
                    )
                    if dist < self.collision_threshold:
                        collision_penalty -= 20.0
                        self.collision_count += 1
            reward += collision_penalty
            
            # 4. Bonus za osiągnięcie celu
            if current_distance < 0.5:
                reward += 50.0
            
            # 5. Mała kara za zbyt wysoką prędkość (zachęca do płynnego ruchu)
            speed = np.linalg.norm(self.robot_velocities[robot_name])
            if speed > 0.8 * self.max_linear_vel:
                reward -= 0.5
            
            # 6. Mała nagroda za ruch (zapobiega staniu w miejscu)
            if speed > 0.1:
                reward += 0.1
            
            rewards[i] = reward
        
        return rewards
    
    def _check_done(self) -> bool:
        """
        Sprawdza czy epizod się zakończył.
        Zakończenie następuje gdy wszystkie roboty osiągną cel.
        """
        all_at_target = True
        for robot_name in self.robot_names:
            distance = np.linalg.norm(
                self.robot_positions[robot_name] - self.target_position
            )
            if distance > 0.5:
                all_at_target = False
                break
        
        if all_at_target:
            self.success_count += 1
        
        return all_at_target
    
    def _detect_collisions(self) -> list:
        """Wykrywa aktualne kolizje między robotami."""
        collisions = []
        for i, robot1 in enumerate(self.robot_names):
            for robot2 in self.robot_names[i+1:]:
                dist = np.linalg.norm(
                    self.robot_positions[robot1] - self.robot_positions[robot2]
                )
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
                
                # Aktualizacja pozycji i orientacji z odometrii
                if data["odom"] is not None:
                    odom = data["odom"]
                    self.robot_positions[robot_name] = np.array([
                        odom.pose.pose.position.x,
                        odom.pose.pose.position.y
                    ])
                    
                    # Konwersja kwaterniona na kąt Eulera (yaw)
                    q = odom.pose.pose.orientation
                    yaw = math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    )
                    self.robot_orientations[robot_name] = yaw
                    
                    # Aktualizacja prędkości
                    self.robot_velocities[robot_name] = np.array([
                        odom.twist.twist.linear.x,
                        odom.twist.twist.linear.y
                    ])
                
                # Aktualizacja danych z lidaru
                if data["scan"] is not None:
                    self.robot_lidar_data[robot_name] = data["scan"]
    
    def get_statistics(self) -> dict:
        """Zwraca statystyki środowiska."""
        return {
            "collision_count": self.collision_count,
            "success_count": self.success_count,
            "average_distance_to_target": np.mean([
                np.linalg.norm(pos - self.target_position)
                for pos in self.robot_positions.values()
            ])
        }