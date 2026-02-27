import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration

from control_framework.config import load_config

def generate_launch_description():
    config = load_config()
    
    env_arg = DeclareLaunchArgument(
        'env',
        default_value='sitl',
        description='Environment to run (sitl or hitl)'
    )
    
    controller_items = config.controllers.sitl.items.keys()
    
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

    return ld