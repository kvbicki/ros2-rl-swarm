import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import xacro

def generate_launch_description():

    gui_arg = DeclareLaunchArgument(
        'gui', default_value='true', description='Enable joint_state_publisher_gui'
    )
    model_arg = DeclareLaunchArgument(
        'model', default_value='robot_3d.urdf.xacro', description='URDF/Xacro model file'
    )
    rviz_arg = DeclareLaunchArgument(
        'rvizconfig',
        default_value=PathJoinSubstitution([FindPackageShare('robot_description'), 'rviz', 'robot.rviz']),
        description='RViz config file'
    )

    def launch_setup(context, *args, **kwargs):
        model_file = os.path.join(
            FindPackageShare('robot_description').perform(context),
            'model', 'robots',
            LaunchConfiguration('model').perform(context)
        )
        robot_desc = xacro.process_file(model_file).toxml()

        rsp_node = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        )

        jsp_node = Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher',
            output='screen'
        )

        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rvizconfig')]
        )

        return [rsp_node, jsp_node, rviz_node]

    return LaunchDescription([gui_arg, model_arg, rviz_arg, OpaqueFunction(function=launch_setup)])
