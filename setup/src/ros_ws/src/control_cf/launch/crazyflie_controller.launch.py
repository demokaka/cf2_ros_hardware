import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch motion_capture_tracking system
    # Get the directory of the motion capture tracking package
    # mocap_tracking_launch_dir = get_package_share_directory('motion_capture_tracking')

    # # Include the motion capture tracking launch file
    # mocap_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(mocap_tracking_launch_dir, 'launch', 'launch.py')
    #     )
    # )
    
    # Launch the Crazyflie controller
    controller_node = Node(
        package='control_cf',
        executable='crazyflie_controller_sequential.py',
        name='crazyflie_controller',
        output='screen'
    )
    
    # cluster_node = Node(
    #     package='crazyflie_control',
    #     executable='cluster_node',
    #     name='cluster_node',
    #     output='screen'
    # )

    ld = LaunchDescription()

    # ld.add_action(mocap_launch)
    
    delay_duration = 2.0
    ld.add_action(TimerAction(
        actions=[controller_node],  
        period=delay_duration
    ))
    
    # delay_duration = 2.0
    # ld.add_action(TimerAction(
    #     actions=[controller_node, cluster_node],  
    #     period=delay_duration
    # ))

    return ld