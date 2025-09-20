import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_urdf_path = get_package_share_directory('robot_description')
    pkg_gazebo_path = get_package_share_directory('robot_gazebo')

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz.'
    )

    world_arg = DeclareLaunchArgument(
        'world', default_value='test_world.sdf',
        description='Name of the Gazebo world file to load'
    )

    model_arg = DeclareLaunchArgument(
        'model', default_value='robot_3d.urdf.xacro',
        description='Name of the URDF description to load'
    )

    urdf_file_path = PathJoinSubstitution([
        pkg_urdf_path,
        "model", "robots",
        LaunchConfiguration('model')
    ])

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_path, 'launch', 'world.launch.py')
        ),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_urdf_path, 'rviz', 'robot.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': True}]
    )

    spawn_nodes = []
    state_publisher_nodes = []
    bridge_nodes = []

    positions = [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.0, 1.0),
        (0.5, 0.25),
        (0.5, 0.75),
        (1, 0.5)
    ]

    for i, (x_pos, y_pos) in enumerate(positions):
        name = f"my_robot_{i+1}"

        spawn_node = Node(
            package="ros_gz_sim",
            executable="create",
            namespace=name,
            arguments=[
                "-name", name,
                "-topic", "robot_description",
                "-x", str(x_pos), "-y", str(y_pos), "-z", "0.5", "-Y", "0.0"
            ],
            output="screen",
            parameters=[{'use_sim_time': True}]
        )

        state_pub_node = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=name,
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': Command(['xacro', ' ', urdf_file_path]),
                'use_sim_time': True
            }]
        )

        bridge_node = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            namespace=name,
            arguments=[
                f"/{name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
                f"/{name}/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
                f"/{name}/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
                f"/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
                f"/{name}/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan"
            ],
            output="screen",
            parameters=[{'use_sim_time': True}]
        )

        spawn_nodes.append(spawn_node)
        state_publisher_nodes.append(state_pub_node)
        bridge_nodes.append(bridge_node)

    clock_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock]"],
        output="screen",
        parameters=[{'use_sim_time': True}]
    )

    ld = LaunchDescription()
    # ld.add_action(rviz_launch_arg)
    # ld.add_action(rviz_node)
    ld.add_action(world_arg)
    ld.add_action(model_arg)
    ld.add_action(world_launch)
    ld.add_action(clock_bridge_node)

    for sn, spn, bn in zip(spawn_nodes, state_publisher_nodes, bridge_nodes):
        ld.add_action(sn)
        ld.add_action(spn)
        ld.add_action(bn)

    return ld
