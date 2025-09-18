from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
from xacro import process_file

def generate_launch_description():

    pkg_robot_description = get_package_share_directory('robot_description')
    urdf_file = os.path.join(pkg_robot_description, 'model', 'robot.xacro')

    doc = process_file(urdf_file)
    robot_desc = doc.toxml()

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        )
    )
    
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'robot1'],
        output='screen'
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )



    return LaunchDescription([
        gazebo_launch,
        robot_state_publisher_node,
        spawn_entity
    ])
