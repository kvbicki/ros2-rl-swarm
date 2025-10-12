#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='ppo',
        description='RL algorithm to use (ppo, sac, dqn)'
    )
    
    training_arg = DeclareLaunchArgument(
        'training',
        default_value='true',
        description='Enable training mode'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/',
        description='Path to save/load models'
    )
    
    swarm_manager_node = Node(
        package='swarm_manager',
        executable='swarm_manager_node.py',
        name='swarm_manager',
        parameters=[{
            'algorithm': LaunchConfiguration('algorithm'),
            'training': LaunchConfiguration('training'),
            'model_path': LaunchConfiguration('model_path')
        }],
        output='screen'
    )
    
    return LaunchDescription([
        algorithm_arg,
        training_arg,
        model_path_arg,
        swarm_manager_node
    ])