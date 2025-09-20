import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution, FindExecutable

def generate_launch_description():
    pkg_gazebo = FindPackageShare("robot_gazebo").find("robot_gazebo")
    pkg_desc = FindPackageShare("robot_description").find("robot_description")

    world_path = os.path.join(pkg_gazebo, "worlds", "test_world.sdf")
    robot_urdf = os.path.join(pkg_desc, "description", "robot1.urdf.xacro")

    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        robot_urdf
    ])

    return LaunchDescription([
        ExecuteProcess(
            cmd=["gz", "sim", "-v4", world_path],
            output="screen"
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen"
        ),

        TimerAction(
            period=2.0,
            actions=[Node(
                package="ros_gz_sim",
                executable="create",
                arguments=["-world", "test_world", "-topic", "robot_description"],
                output="screen"
            )]
        )
    ])
