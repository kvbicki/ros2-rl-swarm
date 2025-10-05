from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='swarm_manager',
            executable='swarm_manager_node.py',
            name='swarm_manager_node',
            output='screen'
        ),
        Node(
            package='swarm_manager',
            executable='rl_agent.py',
            name='rl_agent',
            output='screen'
        ),
        Node(
            package='swarm_manager',
            executable='swarm_env.py',
            name='swarm_env',
            output='screen'
        )
    ])
