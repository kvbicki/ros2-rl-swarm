from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
import os

def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('robot_description'),
        'model',
        'robot.xacro'
    )

    robot_description = Command(['xacro ', urdf_file])

    return LaunchDescription([
        # Publikacja URDF do parametru robot_description
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
        ),
        # Gazebo
        Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_robot',
            arguments=['-param', 'robot_description', '-ros-args'],
            output='screen'
        ),
    ])
