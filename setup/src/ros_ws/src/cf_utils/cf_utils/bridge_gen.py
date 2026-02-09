import yaml

# Define the data
# Clock data
"""
- ros_topic_name: "/clock"
  gz_topic_name: "/clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS
"""
data_clock = [
    {
        "ros_topic_name": "/clock",
        "gz_topic_name": "/clock",
        "ros_type_name": "rosgraph_msgs/msg/Clock",
        "gz_type_name": "gz.msgs.Clock",
        "direction": "GZ_TO_ROS",
    }
]

# # Tf data
# """
# - ros_topic_name: "/tf"
#   gz_topic_name: "/world/empty/pose/info"
#   ros_type_name: "tf2_msgs/msg/TFMessage"
#   gz_type_name: "gz.msgs.Pose_V"
#   direction: GZ_TO_ROS
# """
# data_pose = [
#     {
#         "ros_topic_name": "/tf",
#         "gz_topic_name": "/world/empty/pose/info",
#         "ros_type_name": "tf2_msgs/msg/TFMessage",
#         "gz_type_name": "gz.msgs.Pose_V",
#         "direction": "GZ_TO_ROS",
#     }
# ]

# Odometry data
"""
- ros_topic_name: "/crazyflie_1/odom"
  gz_topic_name: "/model/cf_1/odometry"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS
"""
data_odometry = [
    {
        "ros_topic_name": "/crazyflie_1/odom",
        "gz_topic_name": "/model/cf_1/odometry",
        "ros_type_name": "nav_msgs/msg/Odometry",
        "gz_type_name": "gz.msgs.Odometry",
        "direction": "GZ_TO_ROS",
    }
]

# Camera data
"""
- ros_topic_name: "cf_${i}/camera_info"
  gz_topic_name: "cf_${gz_index}/camera_info"
  ros_type_name: "sensor_msgs/msg/CameraInfo"
  gz_type_name: "ignition.msgs.CameraInfo"
  subscriber_queue: 10
  publisher_queue: 10
  lazy: false
  direction: GZ_TO_ROS

- ros_topic_name: "cf_${i}/image"
  gz_topic_name: "cf_${gz_index}/camera"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "ignition.msgs.Image"
  subscriber_queue: 10
  publisher_queue: 10
  lazy: false
  direction: GZ_TO_ROS
"""

# Battery data
"""
- ros_topic_name: "cf_${i}/battery_status"
  gz_topic_name: "model/crazyflie_${gz_index}/battery/linear_battery/state"
  ros_type_name: "sensor_msgs/msg/BatteryState"
  gz_type_name: "ignition.msgs.BatteryState"
  subscriber_queue: 10
  publisher_queue: 10
  lazy: false
  direction: GZ_TO_ROS
"""
data_battery = [
    {
        "ros_topic_name": "/crazyflie_1/battery_status",
        "gz_topic_name": "/model/cf_1/battery/linear_battery/state",
        "ros_type_name": "sensor_msgs/msg/BatteryState",
        "gz_type_name": "gz.msgs.BatteryState",
        "direction": "GZ_TO_ROS",
        "subscriber_queue": 10,
        "publisher_queue": 10,
        "lazy": False
    }
]

def generate_bridge_config_file(file_path, num_drones, world="empty"):
    """
    Generate a YAML configuration file at runtime with topic mappings.
    """

    # Write to YAML file
    with open(file_path, "w") as file:
        file.write("---\n") # add the '---'
        yaml.dump(data_clock, file, default_flow_style=False, sort_keys=False)

        # data_pose['gz_topic_name'] = f"/world/{world}/pose/info"
        # yaml.dump(data_pose, file, default_flow_style=False, sort_keys=False)

        for i in range(1,num_drones+1):
            indexed_data_odometry = data_odometry.copy()
            indexed_data_odometry[0]['ros_topic_name'] = f"/crazyflie_{i}/odom"
            indexed_data_odometry[0]['gz_topic_name'] = f"/cf_{i}/odom"

            # indexed_data_pose = data_pose.copy()
            # # indexed_data_pose[0]['ros_topic_name'] = f"/crazyflie_{i}/battery_status"
            # indexed_data_pose[0]['gz_topic_name'] = f"/model/cf_{i}/pose"

            indexed_data_battery = data_battery.copy()
            indexed_data_battery[0]['ros_topic_name'] = f"/crazyflie_{i}/battery_status"
            indexed_data_battery[0]['gz_topic_name'] = f"/model/cf_{i}/battery/linear_battery/state"


            yaml.dump(indexed_data_odometry, file, default_flow_style=False, sort_keys=False)
            # yaml.dump(indexed_data_pose, file, default_flow_style=False, sort_keys=False)
            yaml.dump(indexed_data_battery, file, default_flow_style=False, sort_keys=False)

if __name__=="__main__":
    NO_DRONES = 5
    file_path = "bridge_config.yaml"
    generate_bridge_config_file(file_path, NO_DRONES)