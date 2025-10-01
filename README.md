# Reinforcement Learning for Collision Avoidance in Robot Swarms

This project explores the use of **Reinforcement Learning (RL)** to solve the collision avoidance problem in robot swarms.  
It combines **ROS2 Jazzy**, **Gazebo Harmonic**, and **PyTorch** to build a simulation environment where multiple robots learn navigation strategies.

---

## Features
- Multi-robot swarm simulation in **Gazebo Harmonic**  
- Robot communication and control via **ROS2 Jazzy**  
- Reinforcement Learning agents implemented in **PyTorch**  
- Training, evaluation, and analysis of collision avoidance strategies  

---

## Installation

### Requirements
- Ubuntu 24.04  
- ROS2 Jazzy  
- Gazebo Harmonic  
- Python 3.10+  
- PyTorch  


## Goal
The goal of this project is to investigate how RL techniques can improve collision avoidance in swarm robotics and demonstrate their effectiveness in simulation.

## Utils
Key commands for the project:

### Install dependencies

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Build the entire workspace

```bash
colcon build
```

### Build a meta package without entire workspace
```bash
colcon build --packages-up-to swarm_meta
```
### Source the workspace after building
``` bash
source install/setup.bash
```
### Run the teleop and move the robot
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r __ns:=/ -r cmd_vel:=/cmd_vel
```
### Run the teleop and move robot in swarm
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r __ns:=/robot_1 -r cmd_vel:=/robot_1/cmd_vel
```

### Run to see the tf frames
```bash
ros2 run tf2_tools view_frames
```