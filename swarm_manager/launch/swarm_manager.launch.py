#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    default_model_path = os.path.join(os.path.expanduser('~'), 'swarm_models')

    training_arg = DeclareLaunchArgument(
        'training',
        default_value='true',
        description='Enable training mode (true/false)'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='Path to save/load models'
    )
    
    swarm_manager_node = Node(
        package='swarm_manager',
        executable='swarm_manager_node.py',
        name='swarm_manager',
        output='screen',
        parameters=[{
            'training': LaunchConfiguration('training'),
            'model_path': LaunchConfiguration('model_path')
        }],
        prefix=['stdbuf -o L'],
    )
    
    return LaunchDescription([
        training_arg,
        model_path_arg,
        swarm_manager_node
    ])