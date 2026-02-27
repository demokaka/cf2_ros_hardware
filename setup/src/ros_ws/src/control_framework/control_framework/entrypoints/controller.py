from __future__ import annotations
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from control_framework.config.models.global_config import GlobalConfig
from control_framework.config.models.agents import EnvAgentsConfig, CrazyflieConfig
from control_framework.config.models.controllers import ControllerConfig, EnvControllersConfig

from control_framework.crazyflie_descriptor import CrazyflieDescriptor

from control_framework.config import load_config

from control_framework.control_laws.control_law import ControlLaw
from control_framework.control_laws.pid import (
    PIDControlLawSetting, PIDControlLaw
)
from control_framework.control_laws.lqr import (
    LQRControlLawSetting, LQRControlLaw,
)
from control_framework.control_laws.mpc import (
    MPCControlLawSetting, MPCControlLaw
)

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from control_framework.control_laws.models import DoubleIntegrator3DOF

import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Accel

import time

from std_msgs.msg import Float64MultiArray
from rclpy.publisher import Publisher

from motion_capture_tracking_interfaces.msg import NamedPoseArray # type: ignore

import csv # Add this import at the top

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        self.controller_name: str 
        self.agent_id: int
        self.env: str
        self.config: GlobalConfig

        self.declare_parameter("env", "sitl")
        self.env = self.get_parameter("env").value

        self.declare_parameter("controller_name", "cx")
        self.controller_name = self.get_parameter("controller_name").value

        self.agent_config: CrazyflieConfig
        self.controller_config: ControllerConfig

        self.cf: CrazyflieDescriptor
        self.cf_name: str

        self.control_law: ControlLaw
        
        self.publisher_: Publisher
        self.g = 9.81

        # Inside ControllerNode.__init__
        self.log_file = None
        self.csv_writer = None

        filename = f"log_{self.controller_name}_{int(time.time())}.csv"

        self.log_file = open(filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        # Header: Timestamp, Ref(x,y,z), Virtual Inputs(vx,vy,vz), Converted(T,R,P)
        self.csv_writer.writerow([
            'timestamp',
            'rx', 'ry', 'rz',
            'rvx', 'rvy', 'rvz',
            'x', 'y', 'z',
            'vx', 'vy', 'vz',
            'ax', 'ay', 'az', 
            'thrust_pwm', 'roll_deg', 'pitch_deg'
        ])


    def load_configuration(self):
        # Placeholder for configuration loading logic
        self.config = load_config()

        agents_config: EnvAgentsConfig = None
        controllers_config: EnvControllersConfig = None

        if self.env == 'sitl':
            agents_config = self.config.agents.sitl
            controllers_config = self.config.controllers.sitl
        elif self.env == 'hitl':
            agents_config = self.config.agents.hitl
            controllers_config = self.config.controllers.hitl
        else:
            self.get_logger().error(f"Unknown environment '{self.env}'. Expected 'sitl' or 'hitl'.")
            raise ValueError(f"Unknown environment '{self.env}'. Expected 'sitl' or 'hitl'.")
        
        self.controller_config: ControllerConfig = controllers_config.items.get(self.controller_name, None)
        if self.controller_config is None:
            self.get_logger().error(f"Unknown controller {self.controller_name}.")
            raise ValueError(f"Unknown controller {self.controller_name}.")

        self.agent_id = self.controller_config.controlled_agent_id
        self.cf_name = agents_config.id_to_name.get(self.agent_id, None)
        self.agent_config = agents_config.agents.get(self.cf_name, None)
        if self.agent_config is None:
            self.get_logger().error(f"Unknown agent with id: {self.agent_id}.")
            raise ValueError(f"Unknown agent with id: {self.agent_id}.")

        self.get_logger().info(f"{self.agent_config}")

    def initialize_controller(self):
        if self.config is None:
            self.get_logger().error("Configuration not loaded. Call load_configuration() first.")
            raise RuntimeError("Configuration not loaded. Call load_configuration() first.")
        
        ### create agent descriptor
        self.cf = CrazyflieDescriptor(
            name=self.cf_name,
            id=self.agent_id,
            mass=self.agent_config.mass,
            uri=self.agent_config.uri
        )

        ### create control law instance based on type and parameters
        self.__initialize_control_law()

        ### set up ROS publishers/subscribers as needed
        self.__initialize_subscribers()
        self.__initialize_publishers()

        ###
        self.publisher_.publish(Accel())

        ### start the control loop
        self.create_timer(
            timer_period_sec=self.controller_config.parameters.dt / 1000.0,
            callback=self.__control_loop,
        )

    def __control_loop(self):
        t_start = time.perf_counter()

        v = self.control_law.compute_control_action(self.cf)
        thrust, roll, pitch = self.__get_cf_input(v, mass=self.agent_config.mass)
        
        msg = Accel()
        
        if self.cf.trajectory_idx != 0:
            msg.linear.x = float(thrust)
            msg.linear.y = roll
            msg.linear.z = pitch

            # self.csv_writer.writerow([
            #             'timestamp',
            #             'rx', 'ry', 'rz',
            #             'rvx', 'rvy', 'rvz',
            #             'x', 'y', 'z',
            #             'vx', 'vy', 'vz',
            #             'ax', 'ay', 'az', 
            #             'thrust_pwm', 'roll_deg', 'pitch_deg'
            #         ])


        if self.cf.trajectory_idx == 0:
            ref = np.array([0.0, 0.0, 0.0, \
                            0.0, 0.0, 0.0])
        else:
            ref = self.cf.reference_trajectory[self.cf.trajectory_idx]

        curp = self.cf.current_state.position
        curv = self.cf.current_state.velocity
        cura = self.cf.current_state.acceleration

        # --- LOGGING START ---
        if self.csv_writer:
            self.csv_writer.writerow([
                time.time(),
                ref[0], ref[1], ref[2],
                ref[3], ref[4], ref[5],
                curp[0], curp[1], curp[2],
                curv[0], curv[1], curv[2],
                cura[0], cura[1], cura[2],
                v[0], v[1], v[2],
                thrust, roll, pitch
            ])
        # --- LOGGING END ---


        self.publisher_.publish(msg)
        t_end = time.perf_counter()

        # self.get_logger().info(f"Control loop time per iteration: {int((t_end - t_start) * 1_000_000)} us.")

        # self.get_logger().info(fs"Sent control inputs {msg} ref: {ref}, y: {y}")

    def __initialize_subscribers(self):

        if self.env == "sitl":
            self.get_logger().info(f"Initializing subscriber on topic: {self.cf.sitl_odometry_topic}.")

            self.create_subscription(
                Odometry,
                self.cf.sitl_odometry_topic,
                self.cf.sitl_odometry_callback,
                10
            )
            
        elif self.env == "hitl":
            self.get_logger().info(f"Initializing subscriber on topic: {self.cf.qmt_odometry_topic}.")

            self.create_subscription(
                NamedPoseArray,
                self.cf.qmt_odometry_topic,
                self.cf.qmt_odometry_callback,
                QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=QoSReliabilityPolicy.BEST_EFFORT
                )
            )
        
        else:
            return

        self.create_subscription(
            Float64MultiArray,
            self.cf.trajectory_topic,
            self.cf.trajectory_callback,
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT
            )

        )
        self.get_logger().info(f"Initializing subscriber on topic: {self.cf.trajectory_topic}.")

        return
    
    def __initialize_publishers(self):        
        topic_name = f'/{self.cf_name}/control_input'
        
        self.publisher_ = self.create_publisher(
            Accel,
            topic_name,
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT
            )
        )

        self.get_logger().info(f"Publishing on {topic_name} for agent {self.cf_name}: {self.agent_id}")

    def __initialize_control_law(self):
        
        model = DoubleIntegrator3DOF.from_dt(self.controller_config.parameters.dt / 1000.0)
        controller_type = self.controller_config.controller_type

        match controller_type:
            case "PID":
                settings = PIDControlLawSetting(model)
                
                settings.set_gains(
                    Kp=np.array(self.controller_config.parameters.Kp),
                    Ki=np.array(self.controller_config.parameters.Ki),
                    Kd=np.array(self.controller_config.parameters.Kd), 
                )

                settings.set_saturation_limits(
                    u_min=np.array(self.controller_config.parameters.u_min),
                    u_max=np.array(self.controller_config.parameters.u_max)
                )

                self.control_law = PIDControlLaw(settings)

            case "LQR":
                settings = LQRControlLawSetting(model)
                
                settings.set_weights(
                    Q=np.diag(self.controller_config.parameters.Q),
                    R=np.diag(self.controller_config.parameters.R),
                )
                
                settings.set_saturation_limits(
                    u_min=np.array(self.controller_config.parameters.u_min),
                    u_max=np.array(self.controller_config.parameters.u_max)
                )
                
                self.control_law = LQRControlLaw(settings)
                
                return
            case "MPC":
                
                settings = MPCControlLawSetting(model, self.controller_config.parameters.horizon)
                
                settings.set_weights(
                    Q=np.diag(self.controller_config.parameters.Q),
                    R=np.diag(self.controller_config.parameters.R),
                    Qf=np.diag(self.controller_config.parameters.Qf)
                )
                
                settings.set_input_constraints(
                    u_min=np.array(self.controller_config.parameters.u_min),
                    u_max=np.array(self.controller_config.parameters.u_max)
                )
                
                settings.set_state_constraints(
                    x_min=np.array(self.controller_config.parameters.x_min),
                    x_max=np.array(self.controller_config.parameters.x_max)
                )
                
                self.control_law = MPCControlLaw(settings)
                
            case _:
                raise ValueError(f"Unknown controller type '{controller_type}' for controller '{self.controller_name}'.")           


    def __get_cf_input(self, v: np.ndarray, yaw: float = 0.0, mass: float = 30.0) -> list:
        """
        Feedback linearization and PWM mapping.
        Receives virtual inputs v = [vx, vy, vz] ([ax, ay, az]), yaw and mass
        Returns thrust (PWM), roll (deg), pitch (deg)
        """       
        T = np.round(np.sqrt(v[0]**2 + v[1]**2 + (v[2] + self.g)**2), 5)
        phi_d = np.round(np.arcsin((v[0] * np.sin(yaw) - v[1] * np.cos(yaw)) / T), 5)
        theta_d = np.round(np.arctan2(v[0] * np.cos(yaw) + v[1] * np.sin(yaw), v[2] + self.g), 5)

        roll_deg = np.degrees(phi_d)
        pitch_deg = np.degrees(theta_d)

        thrust_g = (T / self.g) * mass
        thrust_pwm = self.__thrust_to_pwm(thrust_g)

        return thrust_pwm, roll_deg, pitch_deg

    def __thrust_to_pwm(self, thrust_g: float) -> int:
        """2nd degree polynomial fit for CF motor thrust."""
        a, b, c = -7.36578582, 1465.72737286, 1098.21200142
        pwm = (a * (thrust_g**2)) + (b * thrust_g) + c
        return int(np.clip(pwm, 0, 65535))

def main(args=None):
    # Initialize rclpy
    rclpy.init(args=args)

    node = ControllerNode()
    node.load_configuration()
    node.initialize_controller()

    try:
        rclpy.spin(node, executor=MultiThreadedExecutor(num_threads=10))
    except KeyboardInterrupt:
        ### nothing to treat
        pass
    finally:
        ### keyboard interrupt destroys everything automatically and all contexts are invalidated...
        if hasattr(node, 'log_file') and node.log_file:
            node.get_logger().info("Closing log file...")
            node.log_file.close()
        pass