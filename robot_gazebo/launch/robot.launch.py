from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    robot_description_path = PathJoinSubstitution([
        FindPackageShare("robot_description"),
        "description",
        "robot1.urdf.xacro"
    ])

    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        robot_description_path
    ])

    return LaunchDescription([
        ExecuteProcess(
            cmd=["gz", "sim", "-v4", "empty.sdf"],
            output="screen"
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen"
        ),

        Node(
            package="ros_gz_sim",
            executable="create",
            arguments=["-world", "empty", "-topic", "robot_description"],
            output="screen"
        ),
    ])
