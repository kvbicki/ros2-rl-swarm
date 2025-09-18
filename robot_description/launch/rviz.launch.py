from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from xacro import process_file

def generate_launch_description():

    pkg_robot_description = get_package_share_directory('robot_description')
    urdf_file = os.path.join(pkg_robot_description, 'model', 'robot.xacro')

    doc = process_file(urdf_file)
    robot_desc = doc.toxml()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    rviz_config_file = os.path.join(pkg_robot_description, 'rviz', 'robot.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        rviz_node
    ])
