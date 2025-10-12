# Work in progress

# Reinforcement Learning for Collision Avoidance in Robot Swarms

This project explores the use of **Reinforcement Learning (RL)** to solve the collision avoidance problem in robot swarms.
It combines **ROS2 Jazzy**, **Gazebo Harmonic**, and **PyTorch** to build a simulation environment where multiple robots learn navigation strategies.

---

## Features

* Multi-robot swarm simulation in **Gazebo Harmonic**
* Robot communication and control via **ROS2 Jazzy**
* Reinforcement Learning agents implemented in **PyTorch**
* Training, evaluation, and analysis of collision avoidance strategies

---

## Installation

### Requirements

* Ubuntu 24.04 (ROS2 Gazebo will run locally)
* Python 3.10 (for Docker RL environment)
* PyTorch nightly with CUDA 13.0

## Goal

The goal of this project is to investigate how RL techniques can improve collision avoidance in swarm robotics and demonstrate their effectiveness in simulation.

## Utils

Key commands for the project:

### Build RL Docker environment

```bash
cd ~/ros2_ws/src/ros2-rl-swarm/swarm_manager
# Build Docker image
docker build -t rl_swarm:latest -f docker/Dockerfile .
```

### Run RL container with GPU access

```bash
docker run --gpus all -it --rm rl_swarm:latest
```

### Verify GPU in container

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Run RL training script inside container

```bash
python3 rl_agent.py
```

### Local ROS2/Gazebo commands (outside Docker)

* Build ROS2 workspace:

```bash
colcon build
```

* Source workspace:

```bash
source install/setup.bash
```

* Run teleop for single robot:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r __ns:=/ -r cmd_vel:=/cmd_vel
```

* Run teleop for robot swarm:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r __ns:=/robot_1 -r cmd_vel:=/robot_1/cmd_vel
```

* View TF frames:

```bash
ros2 run tf2_tools view_frames
```

* Clear Gazebo cache:

```bash
rm -rf ~/.gazebo ~/.ignition ~/.gz
```
