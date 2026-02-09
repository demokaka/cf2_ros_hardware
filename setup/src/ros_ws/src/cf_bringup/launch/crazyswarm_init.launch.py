import os
import yaml
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def _prepare(context, *args, **kwargs):
    rviz_config_file   = LaunchConfiguration("rviz_config_file").perform(context)

    actions = []
    # RViz2
    rviz_node = Node(
        package='rviz2',
        namespace='',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{
                "use_sim_time": True
        }]
    )
    # actions.append(rviz_node)
    actions.append(rviz_node)

    return actions

def generate_launch_description():

    # load crazyflies
    crazyflies_yaml = os.path.join(
        get_package_share_directory('cf_bringup'),
        'config',
        'crazyflie_robots_sim.yaml')
    
    
    crazyswarm2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('crazyflie'), 'launch'), '/launch.py']),
        launch_arguments={'crazyflies_yaml_file': crazyflies_yaml, 'backend': 'cflib', 'mocap': 'False', 'gui': 'False', 'rviz': 'False'}.items()
    )   

    pkg_cf_bringup = get_package_share_directory('cf_bringup')
    default_rviz_config = os.path.join(pkg_cf_bringup, 'config', 'rviz_trc_00.rviz')
    rviz_config_file = DeclareLaunchArgument(
        "rviz_config_file",
        default_value=default_rviz_config,
        description="Absolute path to the RViz config file to load"
    )

    ld =  LaunchDescription()
    ld.add_action(crazyswarm2_launch)
    ld.add_action(rviz_config_file)

    # Add opaque function (dynamic part)
    ld.add_action(OpaqueFunction(function=_prepare))
    return ld