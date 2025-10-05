from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='swarm_manager',
            executable='swarm_manager_node',
            name='swarm_manager_node',
            output='screen'
        ),
        Node(
            package='swarm_manager',
            executable='rl_agent',
            name='rl_agent',
            output='screen'
        ),
        Node(
            package='swarm_manager',
            executable='swarm_env',
            name='swarm_env',
            output='screen'
        )
    ])
