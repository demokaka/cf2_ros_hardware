from __future__ import annotations
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from typing import List, Dict
from time import sleep

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.swarm import Swarm, CachedCfFactory
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Assuming these are your custom message types and config loaders
from geometry_msgs.msg import Accel 
from control_framework.config.models.global_config import GlobalConfig
from control_framework.config.models.agents import EnvAgentsConfig, CrazyflieConfig
from control_framework.config import load_config

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class SwarmNode(Node):
    def __init__(self):
        super().__init__(f'swarm_node')

        self.declare_parameter("env", "sitl")
        self.env = self.get_parameter("env").value

        self.get_logger().info(f"SwarmNode starting, env={self.env}")

        self.swarm: Swarm | None = None
        self.config: GlobalConfig | None = None
        self.agent_settings: EnvAgentsConfig | None = None
        
        self.g = 9.81

    def load_configuration(self):
        """Loads configuration similar to ControllerNode logic."""
        self.config = load_config()
        
        if self.env == 'sitl':
            self.agent_settings = self.config.agents.sitl.agents
        elif self.env == 'hitl':
            self.agent_settings = self.config.agents.hitl.agents
        else:
            self.get_logger().error(f"Unknown environment '{self.env}'.")
            raise ValueError(f"Unknown environment '{self.env}'.")

        self.get_logger().info("Configuration loaded and environment verified.")

    def start_swarm(self):
        """Initializes drivers and opens radio links for the swarm."""
        try:
            cflib.crtp.init_drivers(enable_debug_driver=False)
            factory = CachedCfFactory(rw_cache='./cache')

            uris = []
            for _, agent_config in self.agent_settings.items():
                agent_config: CrazyflieConfig
                uris.append(agent_config.uri)

            self.swarm = Swarm(uris, factory=factory)
            self.swarm.open_links()
            self.swarm.reset_estimators()

            # Wait for parameters in parallel to speed up boot
            self.swarm.parallel_safe(self.__wait_for_param_download)
            self.get_logger().info("Swarm communication online.")
            
            # Setup subscribers AFTER links are open
            self.__setup_subscribers()

        except Exception as e:
            self.get_logger().error(f"Failed to start Crazyflie swarm: {e}")

    def __wait_for_param_download(self, scf: SyncCrazyflie):
        """Blocking wait for individual CF parameter sync."""
        while not scf.cf.param.is_updated:
            sleep(0.1)
        self.get_logger().info(f"Parameters downloaded for {scf.cf.link_uri}")

    def __setup_subscribers(self):
        """
        Creates a subscriber for each URI. 
        Expects topics like: /name/control_input
        """
        for agent_name, agent_config in self.agent_settings.items():
            agent_config: CrazyflieConfig
            uri = agent_config.uri
            topic_name = f'/{agent_name}/control_input'

            self.create_subscription(
                Accel,
                topic_name,
                lambda msg, u=uri: self.__control_callback(msg, u),
                QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=QoSReliabilityPolicy.BEST_EFFORT
                )
            )

            self.get_logger().info(f"Subscribed to {topic_name} for drone {uri}")

    def __control_callback(self, msg: Accel, uri: str):
        """
        Processes incoming acceleration for a specific drone.
        """        
        scf: SyncCrazyflie = self.swarm._cfs[uri]
        if not scf:
            return

        thrust = msg.linear.x
        roll = msg.linear.y
        pitch = msg.linear.z
        
        scf.cf.commander.send_setpoint(
            float(roll),
            float(pitch),
            0.0, # Yaw rate set to 0
            int(thrust)
        )

    def __stop_swarm(self):
        if self.swarm:
            self.swarm.close_links()
            print("[INFO] [swarm_node_sitl]: Swarm links closed.")
        else:
            print("[INFO] [swarm_node_sitl]: Swarm not initialised.")

    def __del__(self):
        print("[INFO] [swarm_node_sitl]: Shutting down swarm node...")
        self.__stop_swarm()


def main(args=None):
    # Initialize rclpy
    rclpy.init(args=args)

    node = SwarmNode()
    node.load_configuration()
    node.start_swarm()

    try:
        rclpy.spin(node, executor=MultiThreadedExecutor(num_threads=10))
    except KeyboardInterrupt:
        ### nothing to treat
        pass
    finally:
        ### keyboard interrupt destroys everything automatically and all contexts are invalidated...
        pass

if __name__ == '__main__':
    main()