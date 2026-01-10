#!/bin/bash
# test_model.sh - Skrypt do testowania wytrenowanego modelu

echo "==================================="
echo "ROS2 Swarm RL Model Testing"
echo "==================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

source install/setup.bash

DEFAULT_MODEL_PATH="/ros2_ws/src/ros2-rl-swarm/swarm_manager/models"
MODEL_PATH=${1:-"$DEFAULT_MODEL_PATH"}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Training mode: false (inference only)"
echo "  Model path: $MODEL_PATH"
echo ""

if [ ! -f "$MODEL_PATH/ppo_best.pth" ]; then
    echo -e "${RED}Error: No trained model found at $MODEL_PATH/ppo_best.pth${NC}"
    echo "Check if the path is correct or if you have run training successfully."
    exit 1
fi

echo -e "${GREEN}Starting Gazebo simulation (Visual Mode)...${NC}"
ros2 launch robot_gazebo spawn_swarm.launch.py headless:=False &
GAZEBO_PID=$!

echo "Waiting for Gazebo (8 seconds)..."
sleep 8

echo -e "${GREEN}Starting Swarm Manager in TEST mode...${NC}"
ros2 launch swarm_manager swarm_manager.launch.py \
    training:=false \
    model_path:=$MODEL_PATH

echo -e "${YELLOW}Shutting down...${NC}"
kill $GAZEBO_PID 2>/dev/null
wait $GAZEBO_PID 2>/dev/null

echo -e "${GREEN}Test completed!${NC}"