import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from string import Template

# Multi-agent Crazyflie SITL launcher (pure ROS 2 launch version)
#
# Pipeline:
#  1) Read YAML (names, poses, ports, etc.)
#  2) Render per-agent SDF from model.sdf.jinja via jinja_gen.py
#  3) Start Gazebo (empty or world)
#  4) Spawn all agents using ros_gz_sim create
#  5) Start ros_gz_bridge to bridge topics/services
#  6) Start gazebo_tf_publisher to publish world->robot TFs
#  7) Start RViz2 with a given config
#  8) Start cf2 SITL processes for each agent
#  9) Start a crazyflie server to handle commands

import sys, yaml, pathlib, subprocess
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.substitutions import FindPackageShare

from cf_utils.bridge_gen import generate_bridge_config_file

def _load_yaml(p): 
    with open(p, "r") as f: 
        return yaml.safe_load(f)
    
def _ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p

def _pose_args(pose):
    vals = list(pose) + [0.0]*(6-len(pose))
    x,y,z,R,P,Y = vals[:6]
    return ["-x",str(x),"-y",str(y),"-z",str(z),"-R",str(R),"-P",str(P),"-Y",str(Y)]

def _render_sdf(py_exec, jinja_script, template_path, base_dir, out_file,
                cf_name, cf_id, cffirm_udp_port, cflib_udp_port):
    cmd = [
        py_exec, jinja_script, template_path,
        # "--base-dir", base_dir,
        base_dir,
        "--output-file", out_file,
        "--cf_name", cf_name,
        "--cf_id", str(cf_id),
        "--cffirm_udp_port", str(cffirm_udp_port),
        "--cflib_udp_port", str(cflib_udp_port),
    ]
    subprocess.run(cmd, check=True)

def _prepare(context, *args, **kwargs):
    name = LaunchConfiguration("name").perform(context)
    actions = [ExecuteProcess(cmd=["echo", f"Launching {name}"], output="screen")]

    pkg_cf_description = get_package_share_directory('cf_description')

    pkg_cf_utils = get_package_share_directory('cf_utils')
    jinja_py    = os.path.join(pkg_cf_utils, "scripts", "jinja_gen.py")
    if not os.path.exists(jinja_py):
        raise FileNotFoundError(f"jinja_gen.py not found: {jinja_py}")
    
    model_variant   = LaunchConfiguration("model_variant").perform(context)   # e.g., crazyflie or crazyflie_thrust_upgrade
    config_file     = LaunchConfiguration("config_file").perform(context)
    out_dir         = LaunchConfiguration("out_dir").perform(context)

    world_file         = LaunchConfiguration("world_file").perform(context)
    rviz_config_file   = LaunchConfiguration("rviz_config_file").perform(context)
    cf2_path      = LaunchConfiguration("cf2_path").perform(context)
    cf2_delay_sec = float(LaunchConfiguration("cf2_delay_sec").perform(context))

    world_name = os.path.splitext(os.path.basename(world_file))[0]

    env_dir       = os.path.join(pkg_cf_description, "models")  # base dir for loader
    # Load the SDF file from "description" package
    sdf_file      = os.path.join(pkg_cf_description, "models", model_variant, "model.sdf.jinja")
    if not os.path.exists(sdf_file):
        raise FileNotFoundError(f"Template not found: {sdf_file}")
    
    _ensure_dir(out_dir)

    if not os.path.exists(cf2_path):
        raise FileNotFoundError(f"cf2 executable not found at: {cf2_path}")

    # Read YAML of agents
    cfg = _load_yaml(config_file)
    agents = cfg.get("crazyflies", [])
    if not agents:
        raise RuntimeError("No 'crazyflies' list in YAML.")
    
    # Render each sdf to /tmp/cf_<name>.sdf (or as requested)
    rendered = []
    for a in agents:
        name = a["name"]
        cf_id = int(a.get("cf_id", 0))
        cffirm_udp = int(a.get("cffirm_udp_port", 19950 + cf_id))
        cflib_udp  = int(a.get("cflib_udp_port",  19850 + cf_id))
        out_file   = os.path.join(out_dir, f"{name}.sdf")
        _render_sdf(sys.executable, jinja_py, sdf_file, env_dir, out_file, 'cf', cf_id, cffirm_udp, cflib_udp)
        print(f"[render_cf_models] wrote: {out_file}")
        rendered.append((a, out_file))

    # Open Gazebo
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    # gz_sim = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
    #     launch_arguments={'gz_args': ['-r -v4 ', 'empty.sdf']}.items(),)
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r -v4 ', f'{world_file}']}.items(),)
    
    actions.append(gz_sim)

    # Spawn all agents with their poses (if provided)
    spawns = []
    for a, sdf_file in rendered:
        name = a["name"]
        pose = a.get("pose", [0.0, 0.0, 0.0, 0, 0, 0])
        spawns.append(
            ExecuteProcess(
                cmd=["ros2","run","ros_gz_sim","create","-name",name,"-file",sdf_file, *_pose_args(pose)],
                output="screen"
            )
        )

    # Slight stagger for stability with many agents
    for i, s in enumerate(spawns):
        actions.append(TimerAction(period=0.3*i, actions=[s]))

    # Use ros_gz_bridge to bridge important topics between ROS and GAZEBO
    NO_DRONES   = len(agents)
    bridge_config_file =  "/tmp/bridge_config.yaml"             # a temporary config file is generated at /tmp/ folder
    generate_bridge_config_file(bridge_config_file, NO_DRONES)   

    bridge_node = Node(                         # use ros_gz_bridge to bridge all the topics and services available
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': bridge_config_file,
        }],
        output='screen'
    )

    actions.append(bridge_node)

    # TF broadcaster for GAZEBO world frame to ROS world frame
    pkg_gazebo_tf_publisher = get_package_share_directory('gazebo_tf_publisher')
    gazebo_tf_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_tf_publisher, 'launch', 'gazebo_tf_publisher_launch.py')),
        launch_arguments={'gz_pose_topic': [f'/world/{world_name}/dynamic_pose/info']}.items(),)
    
    # actions.append(gazebo_tf_publisher)
    actions.append(TimerAction(period=0.3, actions=[gazebo_tf_publisher]))
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
    actions.append(TimerAction(period=0.3, actions=[rviz_node]))

    # Start cf2 SITL processes for each agent
    for i, (agent, _sdf_file) in enumerate(rendered):
        cf_id = int(agent.get("cf_id", 0))
        port  = int(agent.get("cffirm_udp_port", 19950 + cf_id))  # default from cf_id if missing

        cmd = [cf2_path, str(port)]

        # start a bit after spawn to avoid race conditions
        actions.append(
            TimerAction(
                period=cf2_delay_sec + 0.2 * i,
                actions=[ExecuteProcess(cmd=cmd, output="screen")]
            )
        )

    return actions


def generate_launch_description():
    ld = LaunchDescription()
    
    # Declare runtime argument
    name_arg = DeclareLaunchArgument(
        "name",
        default_value="crazyflies",
        description="Name of swarm"
    )
    

    model_variant = DeclareLaunchArgument("model_variant", default_value="crazyflie", 
                                          description="Model folder name under models/")
    
    config_file = DeclareLaunchArgument("config_file", default_value=PathJoinSubstitution([
        FindPackageShare("cf_bringup"), "config", "crazyflie_render.yaml"
        ]), description="YAML with crazyflies list and ports")
    
    out_dir = DeclareLaunchArgument("out_dir", default_value="/tmp", 
                                    description="Output directory for generated SDFs")
    
    pkg_cf_bringup = get_package_share_directory('cf_bringup')
    default_world_file      = os.path.join(pkg_cf_bringup, "worlds", "empty_custom.sdf")
    default_rviz_config = os.path.join(pkg_cf_bringup, 'config', 'rviz_trc.rviz')
    world_file = DeclareLaunchArgument(
        "world_file",
        default_value=default_world_file,
        description="Absolute path to the world SDF to load"
    )
    rviz_config_file = DeclareLaunchArgument(
        "rviz_config_file",
        default_value=default_rviz_config,
        description="Absolute path to the RViz config file to load"
    )

    cf2_path = DeclareLaunchArgument("cf2_path", default_value="/root/crazyflie-firmware/sitl_make/build/cf2",
                          description="Absolute path to cf2 SITL executable")
    cf2_delay_sec = DeclareLaunchArgument("cf2_delay_sec", default_value="0.5",
                                          description="Delay (s) after spawn before starting cf2 per agent")

    ld.add_action(name_arg)
    ld.add_action(model_variant)
    ld.add_action(config_file)
    ld.add_action(out_dir)
    ld.add_action(world_file)
    ld.add_action(rviz_config_file)
    ld.add_action(cf2_path)
    ld.add_action(cf2_delay_sec)
    # Add opaque function (dynamic part)
    ld.add_action(OpaqueFunction(function=_prepare))

    return ld