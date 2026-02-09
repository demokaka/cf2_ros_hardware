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

from functools import partial
import time


class Motors(Enum):
    MOTOR_CLASSIC = 1 # https://store.bitcraze.io/products/4-x-7-mm-dc-motor-pack-for-crazyflie-2 w/ standard props
    MOTOR_UPGRADE = 2 # https://store.bitcraze.io/collections/bundles/products/thrust-upgrade-bundle-for-crazyflie-2-x


class CrazyflieLQRController(Node):
    def __init__(self, rate):
        super().__init__(node_name='cf_lqr_controller')
        self.rate = rate

        config_file = os.path.join(get_package_share_directory('control_cf'), 'config', 'Config_Crazyflie_sim.yaml')
        self.system_parameters = self.load_params(config_file)
        self.no_drones = len(self.system_parameters['drone_bodies'])

        self.is_connected = True

        self.motors = Motors.MOTOR_CLASSIC
        self.position = [None] * (self.no_drones)
        self.velocity = [None] * (self.no_drones)
        self.attitude = [None] * (self.no_drones)

        self.trajectory_changed = True
        self.flight_mode = 'idle'
        self.trajectory_t0 = self.get_clock().now()
        self.current_step = 0

        self.takeoff_duration = 5.0
        self.land_duration = 5.0

        self.roll_int = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.pitch_int = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.thrust_pwm_int = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.roll_p = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.pitch_p = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.thrust_p = {i: 0 for i, _ in enumerate(self.drone_bodies)}


        self.control_queue = [None]
        for i in range(1, self.no_drones + 1):
            self.control_queue.append(deque())

        self.get_logger().info('Initialization completed...')

        self.is_flying = False
        self.lqr_solver = TrackingLQR()

        self.pose_subscribers = {
            i: self.create_subscription(PoseStamped, f'/cf_{i}/pose', partial(self._pose_msg_callback, idx=i-1), 10)
            for i in range(1, self.no_drones + 1)
        }

        self.velocity_subscribers = {
            i: self.create_subscription(LogDataGeneric, f'/cf_{i}/velocity', partial(self._velocity_msg_callback, idx=i-1), 10)
            for i in range(1, self.no_drones + 1)
        }

        self.attitude_setpoint_publishers = {
            i: self.create_publisher(AttitudeSetpoint, f'/cf_{i}/cmd_attitude_setpoint', 10)
            for i in range(1, self.no_drones + 1)
        }

        self.takeoffService = self.create_subscription(Empty, f'/all/lqr_takeoff', self.takeoff, 10)
        self.landService = self.create_subscription(Empty, f'/all/lqr_land', self.land, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/lqr_trajectory', self.start_trajectory, 10)
        self.hoverService = self.create_subscription(Empty, f'/all/lqr_hover', self.hover, 10)

        self.create_timer(self.Ts, self.control_callback)
        self.start_time = time.time()

    def load_params(self, config_file):
        """ Load parameters from config file. """
        with open(config_file) as f:
            system_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.simid = system_parameters['simid']
        self.g = 9.81
        # self.qtm_ip = system_parameters['qtm_ip']
        self.Ts = system_parameters['Ts']
        self.Tsim = system_parameters['Tsim']
        self.m = system_parameters['mass']
        self.uris = system_parameters['uris']
        self.drone_bodies = system_parameters['drone_bodies']
        self.T_coeff = system_parameters['T_coeff']
        self.alpha = system_parameters['alpha']
        self.pwm0 = system_parameters['PWM0']
        self.v_land = 0.01  # m/s
        self.Tto = 10  # Takeoff time in seconds

        return system_parameters
    
    def _pose_msg_callback(self, msg: PoseStamped, idx: int):
        self.position[idx] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.attitude[idx] = tf_transformations.euler_from_quaternion([msg.pose.orientation.x,
                                                                msg.pose.orientation.y,
                                                                msg.pose.orientation.z,
                                                                msg.pose.orientation.w], axes='rxyz')


    def _velocity_msg_callback(self, msg: LogDataGeneric, idx: int):
        self.velocity[idx] = msg.values