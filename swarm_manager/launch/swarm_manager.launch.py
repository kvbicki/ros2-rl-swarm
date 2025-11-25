#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    swarm_manager_node = Node(
        package='swarm_manager',
        executable='swarm_manager_node.py',
        name='swarm_manager',
        parameters=[{
            'training': True,
            'model_path': 'models/'
        }],
        output='screen'
    )
    
    return LaunchDescription([swarm_manager_node])
