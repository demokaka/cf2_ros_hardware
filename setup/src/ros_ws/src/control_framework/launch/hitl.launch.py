import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from control_framework.config import load_config

def generate_launch_description():
    config = load_config()
    
    env_arg = DeclareLaunchArgument(
        'env',
        default_value='hitl',
        description='Environment to run (sitl or hitl)'
    )
    
    controller_items = config.controllers.hitl.items.keys()
    
    ld = LaunchDescription()
    ld.add_action(env_arg)

    swarm_node = Node(
        package='control_framework',
        executable='swarm',
        name='swarm_node',
        parameters=[{'env': LaunchConfiguration('env')}]
    )
    ld.add_action(swarm_node)

    for c_name in controller_items:
        controller_node = Node(
            package='control_framework',
            executable='controller',
            name=f'controller_{c_name}',
            parameters=[{
                'env': LaunchConfiguration('env'),
                'controller_name': c_name
            }],
            remappings=[('__node', f'controller_{c_name}')]
        )

        delayed_start = TimerAction(
            period=15.0,
            actions=[controller_node]
        )
        
        ld.add_action(delayed_start)


    # Launch motion_capture_tracking system
    # Get the directory of the motion capture tracking package
    mocap_tracking_launch_dir = get_package_share_directory('motion_capture_tracking')

    # # Include the motion capture tracking launch file
    mocap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(mocap_tracking_launch_dir, 'launch', 'launch.py')
        )
    )

    ld.add_action(mocap_launch)

    return ld