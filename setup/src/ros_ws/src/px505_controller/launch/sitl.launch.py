import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

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
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="sitl",
        description="Execution mode: sitl or hardware",
    )

    # -------------------------
    # Controller node
    # -------------------------
    controller_node = Node(
        package="px505_controller",          # change if needed
        executable="controller",        # change if needed
        name="controller",
        output="screen",
        arguments=[
            "--mode", LaunchConfiguration("mode"),
        ],
    )
    
    ld = LaunchDescription()
    ld.add_action(mode_arg)

    # ld.add_action(mocap_launch)
    
    delay_duration = 2.0
    ld.add_action(TimerAction(
        actions=[controller_node],  
        period=delay_duration
    ))
    
    return ld