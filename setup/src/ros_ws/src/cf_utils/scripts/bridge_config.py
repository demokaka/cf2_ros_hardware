#!/usr/bin/env python3
import argparse
import yaml
from copy import deepcopy
from pathlib import Path

# ---- helpers ---------------------------------------------------------------

def _m(ros_topic, gz_topic, ros_type, gz_type, direction,
       subscriber_queue=None, publisher_queue=None, lazy=None):
    d = {
        "ros_topic_name": ros_topic,
        "gz_topic_name": gz_topic,
        "ros_type_name": ros_type,
        "gz_type_name": gz_type,
        "direction": direction,
    }
    if subscriber_queue is not None:
        d["subscriber_queue"] = int(subscriber_queue)
    if publisher_queue is not None:
        d["publisher_queue"] = int(publisher_queue)
    if lazy is not None:
        d["lazy"] = bool(lazy)
    return d

def generate_bridge_config_file(
    file_path: str,
    num_drones: int,
    include_actuators: bool = False,
    include_cmd_vel: bool = False,
    include_camera: bool = False,
    include_battery: bool = False,
    # Naming formats
    ros_ns_fmt: str = "crazyflie_{i}",      # ROS robot ns used for odom/cmd topics
    gz_model_fmt: str = "cf_{i}",    # Gazebo <model> name
    camera_ros_ns_fmt: str = "cf_{i}",      # ROS camera ns
    camera_gz_ns_fmt: str = "cf_{i}",       # Gazebo camera ns/topic prefix
    gz_index_offset: int = 0,               # add to i for Gazebo index
    # Queues / behavior for camera+bat
    sub_q: int = 10,
    pub_q: int = 10,
    lazy: bool = False,
):
    """
    Build a ros_gz_bridge parameter YAML.

    NOTE: Keep 'file_path' writable (e.g., /tmp/bridge_config.yaml). Writing
    into installed share/ directories is usually not permitted.
    """

    items = []

    # 0) Clock
    items.append(_m(
        "/clock", "/clock",
        "rosgraph_msgs/msg/Clock", "gz.msgs.Clock", "GZ_TO_ROS"
    ))

    for i in range(num_drones):
        gi = i + gz_index_offset

        ros_ns = ros_ns_fmt.format(i=i)
        gz_model = gz_model_fmt.format(i=gi)

        # 1) Odometry  (ROS: /<ros_ns>/odom, GZ: /model/<gz_model>/odometry)
        items.append(_m(
            f"/{ros_ns}/odom",
            f"/model/{gz_model}/odometry",
            "nav_msgs/msg/Odometry",
            "gz.msgs.Odometry",
            "GZ_TO_ROS"
        ))

    # Write YAML
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("---\n")
        yaml.safe_dump(items, f, default_flow_style=False, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description="Generate ros_gz_bridge config YAML")