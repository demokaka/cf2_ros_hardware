#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class OdomToTF(Node):
    def __init__(self):
        super().__init__('odom_to_tf')
        self.ns = self.declare_parameter('ns', 'cf_1').get_parameter_value().string_value
        self.parent = self.declare_parameter('parent_frame', 'world').get_parameter_value().string_value
        self.child  = self.declare_parameter('child_frame', f'{self.ns}/base_link').get_parameter_value().string_value
        odom_topic  = self.declare_parameter('odom_topic', f'/{self.ns}/odom').get_parameter_value().string_value

        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Odometry, odom_topic, self.cb, 10)
        self.get_logger().info(f'Publishing TF {self.parent} -> {self.child} from {odom_topic}')

    def cb(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id or self.parent
        t.child_frame_id  = self.child
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation      = msg.pose.pose.orientation
        self.br.sendTransform(t)

def main():
    rclpy.init()
    node = OdomToTF()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
