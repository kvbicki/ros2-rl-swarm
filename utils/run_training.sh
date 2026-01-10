#!/bin/bash
# run_training.sh - Skrypt do uruchamiania treningu

echo "==================================="
echo "ROS2 Swarm RL Training Launcher"
echo "==================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ ! -d "install" ]; then
    echo -e "${RED}Error: Workspace not built. Run 'colcon build' first.${NC}"
    exit 1
fi

echo -e "${GREEN}Sourcing workspace...${NC}"
source install/setup.bash

TRAINING=${1:-true}
MODEL_PATH=${2:-"/ros2_ws/src/ros2-rl-swarm/swarm_manager/models"}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Training mode: $TRAINING"
echo "  Model path: $MODEL_PATH"
echo ""

mkdir -p "$MODEL_PATH"

echo -e "${GREEN}Starting Gazebo simulation...${NC}"
ros2 launch robot_gazebo spawn_swarm.launch.py &
GAZEBO_PID=$!

echo "Waiting for Gazebo to start (10 seconds)..."
sleep 10


if ! ps -p $GAZEBO_PID > /dev/null; then
    echo -e "${RED}Error: Gazebo failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}Gazebo started successfully!${NC}"


echo -e "${GREEN}Starting Swarm Manager...${NC}"
ros2 launch swarm_manager swarm_manager.launch.py \
    training:=$TRAINING \
    model_path:=$MODEL_PATH


echo -e "${YELLOW}Shutting down...${NC}"
kill $GAZEBO_PID 2>/dev/null
wait $GAZEBO_PID 2>/dev/null

echo -e "${GREEN}Done!${NC}"