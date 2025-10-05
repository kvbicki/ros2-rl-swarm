#!/usr/bin/env python3
import numpy as np

class SwarmEnv:
    def __init__(self, robot_names):
        self.robot_names = robot_names
        self.num_robots = len(robot_names)

    def reset(self):
        obs = np.zeros((self.num_robots, 3))
        return obs

    def step(self, actions):
        obs = np.random.rand(self.num_robots, 3)
        rewards = np.random.rand(self.num_robots)
        done = False
        info = {}
        return obs, rewards, done, info
