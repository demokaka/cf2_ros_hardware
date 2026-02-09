from px505_controller.interfaces.swarm_interface import SwarmInterface
from px505_controller.control_laws.control_law import ControlLaw
from px505_controller.control_laws.pid import PIDControlLaw
from px505_controller.control_laws.lqr import LQRControlLaw
from px505_controller.control_laws.mpc import MPCControlLaw, make_hover_ref9, make_traj_ref9_window
from px505_controller.interfaces.crazyflie_descriptor import CrazyflieDescriptor, CrazyflieState

from px505_controller.constants import QOSP, CONTROLLER_ACTION, TAKE_OFF_ALTITUDE

from px505_controller.trajectory_generation.utils import flat_poly_path_3d, make_takeoff_3pts, make_landing_3pts, make_circle

from px505_controller.config import load_config, GlobalConfig

from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String

from motion_capture_tracking_interfaces.msg import NamedPoseArray # type: ignore

from ament_index_python.packages import get_package_share_directory

from time import sleep
from copy import deepcopy
from functools import partial

import os
import csv
import numpy as np

class ControllerNode(Node):
    def __init__(self, crazyflies: list[CrazyflieDescriptor], control_laws: dict[int, ControlLaw]):
        super().__init__('controller_node')

        ### Set initial action
        self.current_action = CONTROLLER_ACTION.STAND_BY

        ### Store control laws
        self.control_laws = control_laws
        self.crazyflies = crazyflies
        
        ### Setup CSV Logging

        log_dir = os.path.abspath("/root/ros_ws")
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "flight_log.csv")

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "t", "cf_id",
            "px","py","pz","vx","vy","vz","ax","ay","az",
            "rpx","rpy","rpz","rvx","rvy","rvz","rax","ray","raz",
            "uax","uay","uaz","yaw_cmd"
        ])
        self.csv_file.flush()
        
        self._log_counter = 0
        
        ### Setup Interfaces
        self.swarm_interface = SwarmInterface(
            uris=[c.uri for c in self.crazyflies],
            logger_name=self.get_logger().name
        )                       

        ### Setup ROS Subscribers
        self.create_subscription(String, "/controller/change_action", self.__change_action_callback, QOSP)
                            
        ### Feedback subscriptions
        for cf in self.crazyflies:
            self.create_subscription(
                Odometry,
                cf.odometry_topic,
                cf.odometry_callback,
                QOSP
            )


        ###     
        self.get_logger().info("Controller initialized.")
        
        ### Start Main Loop
        self.__main()
        
    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(f"Saved log to {self.csv_path}")
        except Exception:
            pass
        super().destroy_node()
    
        
    def __main(self):
        
        self.swarm_interface.start()
        self.get_logger().info("Main entry point started.")

        # create timer for control loop
        self.get_logger().info(f"Starting control loop...\n Current action: {self.current_action}")

        for agent_id, control_law in self.control_laws.items():
            self.get_logger().info(f"Control law for agent {agent_id}: {type(control_law).__name__}")
            self.create_timer(
                control_law.settings.model.dt,
                partial(self.__control_iteration, control_law, agent_id)
            )

    def __change_action_callback(self, msg: String):
        new_action = msg.data.strip().lower()

        valid_actions = {
            CONTROLLER_ACTION.STAND_BY,
            CONTROLLER_ACTION.TRACK,
            CONTROLLER_ACTION.HOVER,
            CONTROLLER_ACTION.TAKE_OFF,
            CONTROLLER_ACTION.LAND,
            CONTROLLER_ACTION.EXIT,
        }

        if new_action not in valid_actions:
            self.get_logger().warn(
                f"Unknown action '{new_action}'. Valid: {sorted(list(valid_actions))}"
            )
            return

        self.get_logger().info(f"Changing action from {self.current_action} to {new_action}.")

        match new_action:
            case CONTROLLER_ACTION.STAND_BY:
                self.current_action = CONTROLLER_ACTION.STAND_BY

                for cf in self.crazyflies:
                    self.control_laws[cf.id].reset()

                return
            
            case CONTROLLER_ACTION.TRACK:
                for cf in self.crazyflies:
                    if cf.current_state.position is None:
                        self.get_logger().error(f"Crazyflie {cf.id} position is unknown. Cannot take off.")
                        return
                    # self.control_laws[cf.id].reset()
                    
                    if cf.reference_state.position is None:
                        cf.reference_state.position = cf.current_state.position.copy()
                                        
                    points = np.array([
                        [cf.reference_state.position[0], cf.reference_state.position[1], cf.reference_state.position[2]],
                        [cf.reference_state.position[0] + 1.0, cf.reference_state.position[1] + 1.0, cf.reference_state.position[2] + 0.0],
                        [cf.reference_state.position[0] + 1.0, cf.reference_state.position[1] + -1.0, cf.reference_state.position[2] + 0.0],
                        [cf.reference_state.position[0] + 0.0, cf.reference_state.position[1] + 0.0, cf.reference_state.position[2] + 0.0],
                    ])
                    
                    v_points = np.array([
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ])
                    
                    a_points = np.array([
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ])
                    
                    dt = self.control_laws[cf.id].settings.model.dt
                    Tseg = 10  # seconds per segment (choose something reasonable for track)
                    
                    ts, p_ref, v_ref, a_ref = flat_poly_path_3d(
                        points=points,
                        T_per_segment=Tseg,
                        dt=dt,
                        v_points=v_points,
                        a_points=a_points,
                    )
                    
                    cf.reference_trajectory = []
                    for i in range(p_ref.shape[1]):
                        state = CrazyflieState()
                        state.position = p_ref[:,i]
                        state.velocity = v_ref[:,i]
                        state.acceleration = a_ref[:,i]
                        cf.reference_trajectory.append(state)
                    
                    cf.trajectory_idx = 0
                    
                    # cf.reference_state.position = cf.current_state.position.copy()
                    # cf.reference_state.position[2] = TAKE_OFF_ALTITUDE
                    # cf.reference_state.velocity = np.zeros((3,))

            case CONTROLLER_ACTION.HOVER:
                for cf in self.crazyflies:
                    if cf.current_state.position is None:
                        self.get_logger().error(f"Crazyflie {cf.id} position is unknown. Cannot hover.")
                        return
                    # self.control_laws[cf.id].reset()
                    
                    if cf.reference_state.position is None:
                        cf.reference_state.position = cf.current_state.position.copy()
                        
                    cf.reference_state.velocity = np.zeros((3,))

            case CONTROLLER_ACTION.TAKE_OFF:
                for cf in self.crazyflies:
                    if cf.current_state.position is None:
                        self.get_logger().error(f"Crazyflie {cf.id} position is unknown. Cannot take off.")
                        return
                    self.control_laws[cf.id].reset()
                    
                    points, v_points, a_points = make_takeoff_3pts(
                        p0=cf.current_state.position,
                        z_hover=TAKE_OFF_ALTITUDE,
                        dz=0.0
                    )
                    
                    v_points[1, :] = np.array([0.0, 0.0, 0.2])  # small upward velocity at mid-point

                    dt = self.control_laws[cf.id].settings.model.dt
                    Tseg = 5.0  # seconds per segment (choose something reasonable for takeoff)

                    ts, p_ref, v_ref, a_ref = flat_poly_path_3d(
                        points=points,
                        T_per_segment=Tseg,
                        dt=dt,
                        v_points=v_points,
                        a_points=a_points,
                    )
                                        
                    cf.reference_trajectory = []
                    for i in range(p_ref.shape[1]):
                        state = CrazyflieState()
                        state.position = p_ref[:,i]
                        state.velocity = v_ref[:,i]
                        state.acceleration = a_ref[:,i]
                        cf.reference_trajectory.append(state)
                    
                    cf.trajectory_idx = 0
                    
                    # cf.reference_state.position = cf.current_state.position.copy()
                    # cf.reference_state.position[2] = TAKE_OFF_ALTITUDE
                    # cf.reference_state.velocity = np.zeros((3,))

            case CONTROLLER_ACTION.LAND:
                for cf in self.crazyflies:
                    if cf.current_state.position is None:
                        self.get_logger().error(f"Crazyflie {cf.id} position is unknown. Cannot land.")
                        return
                    # self.control_laws[cf.id].reset()
                    
                    if cf.reference_state.position is None:
                        cf.reference_state.position = cf.current_state.position.copy()
                                        
                    points, v_points, a_points = make_landing_3pts(
                        p_hover=cf.reference_state.position,
                    )
                                        
                    v_points[1, :] = np.array([0.0, 0.0, -0.2])

                    dt = self.control_laws[cf.id].settings.model.dt
                    Tseg = 5.0  # seconds per segment (choose something reasonable for takeoff)

                    ts, p_ref, v_ref, a_ref = flat_poly_path_3d(
                        points=points,
                        T_per_segment=Tseg,
                        dt=dt,
                        v_points=v_points,
                        a_points=a_points,
                    )
                    
                    cf.reference_trajectory = []
                    for i in range(p_ref.shape[1]):
                        state = CrazyflieState()
                        state.position = p_ref[:,i]
                        state.velocity = v_ref[:,i]
                        state.acceleration = a_ref[:,i]
                        cf.reference_trajectory.append(state)
                    
                    cf.trajectory_idx = 0
                    
                    # cf.reference_state.position = cf.current_state.position.copy()
                    # cf.reference_state.position[2] = 0
                    # cf.reference_state.velocity = np.zeros((3,))

            case CONTROLLER_ACTION.EXIT:
                return

        self.current_action = new_action

                
    def __control_iteration(self, control_law: ControlLaw, agent_id: int):   

        # Do nothing if in stand by
        if self.current_action == CONTROLLER_ACTION.STAND_BY:
            return
        
        # Solve control law based on current action
        match self.current_action:
            case CONTROLLER_ACTION.TRACK | CONTROLLER_ACTION.HOVER | CONTROLLER_ACTION.TAKE_OFF | CONTROLLER_ACTION.LAND:
                self.__compute_control_inputs(self.current_action, control_law, agent_id)
            case CONTROLLER_ACTION.EXIT:
                self.get_logger().info("Exiting control loop.")
            case _:
                self.get_logger().warn(f"Unknown action: {self.current_action}. Switching to STAND_BY.")
                self.current_action = CONTROLLER_ACTION.STAND_BY
    
    def __compute_control_inputs(self, action: str, control_law: ControlLaw, agent_id: int):
        """
        Compute and send control inputs for a specific agent using the given control law.
        
        Args:
            action: Current controller action (TAKE_OFF, LAND, TRACK, or HOVER)
            control_law: The control law instance for this agent
            agent_id: The agent/crazyflie ID to compute control for
        """
        # Find the crazyflie descriptor for this agent
        cf : CrazyflieDescriptor = next((cfly for cfly in self.crazyflies if cfly.id == agent_id), None)
        if cf is None:
            self.get_logger().warn(f"Crazyflie with ID {agent_id} not found.")
            return
        
        # Determine if action uses trajectory or hover reference
        uses_trajectory = action in [CONTROLLER_ACTION.TAKE_OFF, CONTROLLER_ACTION.LAND, CONTROLLER_ACTION.TRACK]
        
        # Handle trajectory-based actions
        if uses_trajectory:
            if cf.trajectory_idx >= len(cf.reference_trajectory):
                self.get_logger().info(
                    f"Crazyflie {cf.id} reached end of {action} trajectory. "
                    f"Trajectory had {len(cf.reference_trajectory)} waypoints, "
                    f"completed {cf.trajectory_idx} iterations."
                )
                if cf.reference_trajectory:
                    cf.reference_state.position = cf.reference_trajectory[-1].position
                
                # Transition to next action
                next_action = {
                    CONTROLLER_ACTION.TAKE_OFF: CONTROLLER_ACTION.HOVER,
                    CONTROLLER_ACTION.LAND: CONTROLLER_ACTION.STAND_BY,
                    CONTROLLER_ACTION.TRACK: CONTROLLER_ACTION.HOVER,
                }.get(action, CONTROLLER_ACTION.STAND_BY)
                
                self.__change_action_callback(String(data=next_action))
                return
            
            ref = cf.reference_trajectory[cf.trajectory_idx]
            cf.trajectory_idx += 1
        else:
            # Hover action
            ref = CrazyflieState(
                position=cf.reference_state.position,
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
            )
        
        # Build reference and current states based on control law type
        reference_state, current_state = self.__build_control_states(
            control_law, cf, ref if uses_trajectory else None
        )
        
        if reference_state is None or current_state is None:
            return
        
        # Compute control action
        u = control_law.compute_control_action(current_state, reference_state)
        
        # Log
        self.__log_tick(cf, ref, u, yaw_cmd=0.0)
        
        # Send control input for this agent
        inputs = {
            "ax": u[0],
            "ay": u[1],
            "az": u[2],
            "yaw": 0.0,
            "mass": cf.mass
        }

        self.swarm_interface.send_control_inputs({cf.uri: [inputs]})
    
    def __build_control_states(self, control_law: ControlLaw, cf: CrazyflieDescriptor, 
                               ref: CrazyflieState | None) -> tuple:
        """
        Build reference and current state vectors based on control law type.
        
        Returns:
            Tuple of (reference_state, current_state) or (None, None) on error
        """
        if isinstance(control_law, PIDControlLaw):
            # PID works on position only
            if ref:
                reference_state = ref.position
            else:
                reference_state = cf.reference_state.position
            current_state = cf.current_state.position
        
        elif isinstance(control_law, LQRControlLaw):
            # LQR uses full state [pos; vel]
            if ref:
                reference_state = np.hstack([ref.position, ref.velocity])
            else:
                reference_state = np.hstack([
                    cf.reference_state.position,
                    cf.reference_state.velocity,
                ])
            current_state = np.hstack([
                cf.current_state.position,
                cf.current_state.velocity,
            ])
        
        elif isinstance(control_law, MPCControlLaw):
            # MPC uses trajectory window or hover reference
            if ref:
                reference_state = make_traj_ref9_window(
                    reference_trajectory=cf.reference_trajectory,
                    idx0=cf.trajectory_idx - 1,
                    Npred=control_law.settings.Npred
                )
            else:
                reference_state = make_hover_ref9(pos=cf.reference_state.position)
            current_state = current_state_6(cf)
        
        else:
            self.get_logger().warn(f"Unknown control law for CF {cf.id}, skipping.")
            return None, None
        
        return reference_state, current_state

    def __log_tick(self, cf: CrazyflieDescriptor, ref: CrazyflieState, u: np.ndarray, yaw_cmd: float = 0.0):
        # time in seconds (ROS clock)
        t = self.get_clock().now().nanoseconds * 1e-9

        # current
        p = cf.current_state.position if cf.current_state.position is not None else np.zeros(3)
        v = cf.current_state.velocity if cf.current_state.velocity is not None else np.zeros(3)
        a = cf.current_state.acceleration if cf.current_state.acceleration is not None else np.zeros(3)

        # reference (make sure fields exist)
        rp = ref.position if ref is not None and ref.position is not None else np.zeros(3)
        rv = ref.velocity if ref is not None and ref.velocity is not None else np.zeros(3)
        ra = ref.acceleration if ref is not None and getattr(ref, "acceleration", None) is not None else np.zeros(3)

        u = np.asarray(u, float).reshape(-1)
        uax, uay, uaz = float(u[0]), float(u[1]), float(u[2])

        self.csv_writer.writerow([
            t, cf.id,
            p[0], p[1], p[2], v[0], v[1], v[2], a[0], a[1], a[2],
            rp[0], rp[1], rp[2], rv[0], rv[1], rv[2], ra[0], ra[1], ra[2],
            uax, uay, uaz, float(yaw_cmd)
        ])
        
        self._log_counter += 1
        if self._log_counter % 50 == 0:   # flush every 50 rows
            self.csv_file.flush()
            
            
def current_state_6(cf: CrazyflieDescriptor) -> np.ndarray:
    return np.hstack([cf.current_state.position, cf.current_state.velocity])