import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution, PythonExpression

def generate_launch_description():

    world_arg = DeclareLaunchArgument(
        'world', default_value='test_world.sdf',
        description='Name of the Gazebo world file to load'
    )

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='False',
        description='Run Gazebo in headless mode (no GUI)'
    )

    pkg_robot_gazebo = get_package_share_directory('robot_gazebo')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    gz_flags = PythonExpression([
        '" -r -s -v -v1" if "true" == "', LaunchConfiguration('headless'), '".lower() else " -r -v -v1"'
    ])

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={'gz_args': [PathJoinSubstitution([
            pkg_robot_gazebo,
            'worlds',
            LaunchConfiguration('world')
        ]),
        gz_flags], 
        'on_exit_shutdown': 'true'}.items()
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(headless_arg)
    launchDescriptionObject.add_action(gazebo_launch)

    return launchDescriptionObject