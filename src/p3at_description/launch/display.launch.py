# Copyright 2022 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import LogInfo


def generate_launch_description():

    description_file = os.path.join(get_package_share_directory("p3at_description"), "urdf", "pioneer1.urdf")
    #World file for sim (commented out)
    world_file = os.path.join(get_package_share_directory("p3at_description"), "world", "test1.sdf")

    robot_description = ParameterValue(Command(['xacro ', description_file]), value_type=str)


    #Odom transform node
    odom_transform = Node(
        package='odom_transform',
        executable='odom_transform',
        parameters=[{'use_sim_time': False}]
    )        

    #Phidget node
       #Odom transform node
    phidget = Node(
        package='phidget',
        executable='phidget',
        parameters=[{'use_sim_time': False}]
    )

    joystick_library = Node(
            package='joy_linux',
            executable='joy_linux_node',
            name='joy_publisher')
    
    master = Node(
            package='master',
            executable='master',
            name='master')  
    
    declare_slam_params = DeclareLaunchArgument(
            'slam_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('p3at_description'), 'config', 'mapper_params_online_async.yaml'
            ]),
            description='Full path to the slam_toolbox parameter file.'
        )

    declare_nav2_params = DeclareLaunchArgument(
            'nav2_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('p3at_description'), 'config', 'nav2_params.yaml'
            ]),
            description='Full path to the nav2 parameter file.'
    )
    
    declare_use_sim_time = DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Do not use simulation time'
        )

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz.'
    )

    #log_slam_param = LogInfo(msg=['Using SLAM param file: ', LaunchConfiguration('slam_params_file')])


    # Include the SLAM Toolbox launch file
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'), 'launch', 'online_async_launch.py'
            ])
        ]),
        launch_arguments={
            'slam_params_file': './src/p3at_description/config/mapper_params_online_async.yaml',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items()
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'params_file': './src/p3at_description/config/nav2_params.yaml',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items()
    )

    # Including the lakibeam1 driver launch file correctly
    richbeam_package = IncludeLaunchDescription(
            PythonLaunchDescriptionSource('/workspace/src/Lakibeam_ROS2_Driver/launch/lakibeam1_scan.launch.py')
        )

    phidget_drivers = IncludeLaunchDescription(
            PythonLaunchDescriptionSource('/workspace/src/p3at_description/launch/spatial.launch.py')
        )

    #Manual setup that works (kinda)
    #ros2 launch nav2_bringup navigation_launch.py params_file:=./src/p3at_description/config/nav2_params.yaml use_sim_time:=true
    #ros2 launch slam_toolbox online_async_launch.py slam_params_file:=./src/p3at_description/config/mapper_params_online_async.yaml use_sim_time:=true

    # Get the parser plugin convert sdf to urdf using robot_description topic
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': False}
        ]
    )

    # Launch rviz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory("p3at_description"), 'rviz', 'display.rviz')],
        parameters=[{"use_sim_time" : False}]
    )

    sdf_world = ExecuteProcess(
        cmd=["gz", "sim", world_file],
        name="create_world",
        output="both"
    )

    robot = ExecuteProcess(
        cmd=["ros2", "run", "ros_gz_sim", "create", "-file", description_file, "-z", "0.2"],
        name="spawn robot",
        output="both"
    )

    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{"use_sim_time" : False}]
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
                   "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
                   "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
                   "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
                   "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
                   "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ]
    )

    # A gui tool for easy tele-operation.
    robot_steering = Node(
        package="rqt_robot_steering",
        executable="rqt_robot_steering",
    )

    return LaunchDescription([
        #log_slam_param,
        phidget_drivers,
        joystick_library,
        master,
        richbeam_package,
        declare_use_sim_time, #Declare that we are not using sim time
        declare_nav2_params,
        nav2_launch,
        phidget,
        declare_slam_params,
        odom_transform,
        slam_launch, #Launch slam
        rviz_launch_arg,
        #sdf_world,
        #robot,  #Gazebo sim
        robot_state_publisher,
        #ros_gz_bridge,
        joint_state_pub,
        rviz,
        #robot_steering
    ])