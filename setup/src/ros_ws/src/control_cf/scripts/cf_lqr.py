import os

import rclpy
import rclpy.node
from rclpy.node import Node
from rclpy import executors
from crazyflie_py import *

from ament_index_python.packages import get_package_share_directory
from crazyflie_interfaces.msg import LogDataGeneric, AttitudeSetpoint
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
import tf_transformations


import pathlib
from enum import Enum
# from copy import copy
from collections import deque
import numpy as np
import yaml

from control_cf.tracking_lqr import TrackingLQR

class Motors(Enum):
    MOTOR_CLASSIC = 1 # https://store.bitcraze.io/products/4-x-7-mm-dc-motor-pack-for-crazyflie-2 w/ standard props
    MOTOR_UPGRADE = 2 # https://store.bitcraze.io/collections/bundles/products/thrust-upgrade-bundle-for-crazyflie-2-x


class CrazyflieLQR(Node):
    def __init__(self, cf_name: str, rate):
        super().__init__(node_name='cf_lqr', namespace=cf_name)

        prefix = '/' + cf_name
        
        self.is_connected = True
        self.rate = rate
        self.motors = Motors.MOTOR_CLASSIC
        self.position = []
        self.velocity = []
        self.attitude = []

        self.trajectory_changed = True
        self.flight_mode = 'idle'
        self.trajectory_t0 = self.get_clock().now()

        self.takeoff_duration = 5.0
        self.land_duration = 5.0

        self.control_queue = None
        self.get_logger().info('Initialization completed...')

        self.is_flying = False
        self.lqr_solver = TrackingLQR()

        
        self.create_subscription(
            PoseStamped,
            f'{prefix}/pose',
            self._pose_msg_callback,
            10)
        
        self.create_subscription(
            LogDataGeneric,
            f'{prefix}/velocity',
            self._velocity_msg_callback,
            10)

        self.attitude_setpoint_pub = self.create_publisher(
            AttitudeSetpoint,
            f'{prefix}/cmd_attitude_setpoint',
            10)

        
        self.takeoffService = self.create_subscription(Empty, f'/all/lqr_takeoff', self.takeoff, 10)
        self.landService = self.create_subscription(Empty, f'/all/lqr_land', self.land, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/lqr_trajectory', self.start_trajectory, 10)
        self.hoverService = self.create_subscription(Empty, f'/all/lqr_hover', self.hover, 10)

        self.create_timer(1./self.rate, self._main_loop)
        self.create_timer(1./10, self.lqr_loop)


    def _pose_msg_callback(self, msg: PoseStamped):
        self.position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.attitude = tf_transformations.euler_from_quaternion([msg.pose.orientation.x,
                                                                msg.pose.orientation.y,
                                                                msg.pose.orientation.z,
                                                                msg.pose.orientation.w], axes='rxyz')


    def _velocity_msg_callback(self, msg: LogDataGeneric):
        self.velocity = msg.values                                            

    def cmd_attitude_setpoint(self, roll, pitch, yaw_rate, thrust_pwm):
        setpoint = AttitudeSetpoint()
        setpoint.roll = roll
        setpoint.pitch = pitch
        setpoint.yaw_rate = yaw_rate
        setpoint.thrust = thrust_pwm
        self.attitude_setpoint_pub.publish(setpoint)

    def thrust_to_pwm(self, collective_thrust: float) -> int:
        # omega_per_rotor = 7460.8*np.sqrt((collective_thrust / 4.0))
        # pwm_per_rotor = 24.5307*(omega_per_rotor - 380.8359)
        collective_thrust = max(collective_thrust, 0.) #  make sure it's not negative
        if self.motors == Motors.MOTOR_CLASSIC:
            return int(max(min(24.5307*(7460.8*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))
        elif self.motors == Motors.MOTOR_UPGRADE:
            return int(max(min(24.5307*(6462.1*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))
    
    def takeoff(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'takeoff'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        1.0])

    def hover(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'hover'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        self.position[2]])

    def land(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'land'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        0.1])        

    def start_trajectory(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'trajectory'            

    

    def _main_loop(self):
        if self.flight_mode == 'idle':
            return

        if not self.position or not self.velocity or not self.attitude:
            self.get_logger().warning("Empty state message.")
            return
        
        if not self.is_flying:
            self.is_flying = True
            self.cmd_attitude_setpoint(0.,0.,0.,0)

        if self.control_queue is not None:
            control = self.control_queue.popleft()
            thrust_pwm = self.thrust_to_pwm(control[3])
            yawrate = -3.*(self.attitude[2])
            self.cmd_attitude_setpoint(control[0], 
                                       control[1], 
                                       yawrate, 
                                       thrust_pwm)
    def lqr_loop(self):
        if not self.is_flying:
            return
        
        if self.trajectory_changed:
            self.trajectory_start_position = self.position
            self.trajectory_t0 = self.get_clock().now()
            self.trajectory_changed = False
        
        t = (self.get_clock().now() - self.trajectory_t0).nanoseconds / 10.0**9

        u = np.array([0.,0.,0.])
        self.control_queue = deque(u)


def main():
    control_update_rate = 20  # Hz                 
    n_agents = 2
    
    nodes = []
    for i in range(1, n_agents+1):
        nodes.append(CrazyflieLQR('cf_' + str(i), control_update_rate))
    
    executor = executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
        try:
            while rclpy.ok():
                node.get_logger().info('Beginning multiagent executor, shut down with CTRL-C')
                executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info('Keyboard interrupt, shutting down.\n')
    
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()