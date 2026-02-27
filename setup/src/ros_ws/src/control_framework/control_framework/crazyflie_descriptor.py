from __future__ import annotations

import numpy as np
import time
from typing import List
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from motion_capture_tracking_interfaces.msg import NamedPose, NamedPoseArray  # type: ignore
from std_msgs.msg import Float64MultiArray


class CrazyflieState:
    """
    State representation for a Crazyflie drone.
    Attributes:
        position (np.ndarray): Current position of the Crazyflie.
        velocity (np.ndarray): Current velocity of the Crazyflie.
        acceleration (np.ndarray): Current acceleration of the Crazyflie.
    """
    # Using __slots__ for faster attribute access and lower memory overhead
    __slots__ = ['position', 'velocity', 'acceleration']

    def __init__(self,
                 position: np.ndarray = None,
                 velocity: np.ndarray = None,
                 acceleration: np.ndarray = None):

        # Pre-allocate or use provided arrays to avoid frequent re-allocation
        self.position = position if position is not None else np.zeros(3)
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.acceleration = acceleration if acceleration is not None else np.zeros(3)


class CrazyflieDescriptor:
    """
    Descriptor for a Crazyflie drone.
    Attributes:
        name (str): Name of the Crazyflie chasis. Usually coming from the feedback provider (e.g. motion capture system).
        id (str): Unique identifier for the Crazyflie.
        uri (str): URI of the Crazyflie.
        current_state (CrazyflieState): Current state of the Crazyflie.
        reference_state (CrazyflieState): Reference state for the Crazyflie.
        battery_voltage (float): Current battery voltage of the Crazyflie.

        sitl_odometry_topic (str): ROS topic for SITL odometry data.
        battery_topic (str): ROS topic for battery data.
        qmt_odometry_topic (str): ROS topic for QMT odometry data.
        trajectory_topic (str): ROS topic for reference trajectory data.
    """

    def __init__(self, name: str, id: int, mass: float, uri: str):
        self.name = name
        self.id = id
        self.mass = mass
        self.uri = uri

        self.current_state = CrazyflieState()
        self.reference_state = CrazyflieState()
        self.trajectory_idx = 0

        # Internal buffers for acceleration estimation
        self._last_vel = np.zeros(3)
        self._last_time = None

        # Reference trajectory stored as (N, 9) matrix: [pos(3), vel(3), acc(3)]
        self.reference_trajectory = np.empty((0, 9))
        self.hold_reference = np.zeros((9,), dtype=float)
        self._hold_initialized = False
        self.battery_voltage = 0.0

        # Pre-compute strings
        self.sitl_odometry_topic = f"/{name}/odom"
        self.battery_topic = f"/{name}/battery_status"
        self.qmt_odometry_topic = f"/poses"
        self.trajectory_topic = f"/{name}/trajectory"

        self.print_count = 20

    def _estimate_acceleration(self, current_vel: np.ndarray):
        """ Simple raw estimation: a = (v_now - v_prev) / dt """
        now = time.time()

        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                # Direct numerical differentiation
                self.current_state.acceleration[:] = (current_vel - self._last_vel) / dt

        self._last_vel[:] = current_vel
        self._last_time = now

    def _parse_traj_msg(self, msg: Float64MultiArray) -> np.ndarray:
        """
        Parse Float64MultiArray into traj array (N, 9) with row-major convention.

        Expected convention from the trajectory node:
          - layout.dim[0] = points
          - layout.dim[1] = features (must be 9)
          - data is row-major flattened (N*9 values)
        """
        data = np.array(msg.data, dtype=float).reshape(-1,)
        if data.size == 0:
            return np.empty((0, 9), dtype=float)

        n_points = None
        n_features = None

        if msg.layout is not None and msg.layout.dim is not None and len(msg.layout.dim) >= 2:
            try:
                n_points = int(msg.layout.dim[0].size)
                n_features = int(msg.layout.dim[1].size)
            except Exception:
                n_points = None
                n_features = None

        if n_points is None or n_features is None:
            if data.size % 9 != 0:
                raise ValueError(f"trajectory data length must be multiple of 9, got {data.size}")
            n_points = int(data.size // 9)
            n_features = 9

        if n_features != 9:
            raise ValueError(f"expected 9 features per point, got {n_features}")

        expected = n_points * n_features
        if data.size != expected:
            raise ValueError(f"layout expects {expected} floats, got {data.size}")

        return data.reshape((n_points, n_features))

    def trajectory_callback(self, msg: Float64MultiArray):
        """
        ROS 2 callback to receive a full reference trajectory on /<agent>/trajectory.
        """
        try:
            traj = self._parse_traj_msg(msg)
        except Exception:
            # keep silent here; controller node can log if desired
            return

        if traj.size == 0:
            return

        self.set_reference_trajectory(traj)

    def get_reference_point(self) -> np.ndarray:
        """
        Retrieves the next reference point from the trajectory as a (9,) vector.

        Workflow:
        - If no trajectory: return hold_reference
        - Consume one point per call until the last point
        - When last point reached: keep returning the last point forever
        """
        traj = self.reference_trajectory
        n_total = traj.shape[0]

        if n_total == 0:
            return self.hold_reference

        i = int(self.trajectory_idx)
        if i < 0:
            i = 0
        if i >= n_total:
            i = n_total - 1

        ref = traj[i, :]

        if self.trajectory_idx < n_total - 1:
            self.trajectory_idx += 1
        else:
            self.trajectory_idx = n_total - 1
            self.hold_reference[:] = traj[-1, :]

        return ref

    def get_mpc_window(self, n_pred: int) -> np.ndarray:
        """
        Returns (n_pred, 9) window for MPC and advances trajectory by 1 step.

        - If no trajectory: repeats hold_reference
        - Pads with last point if needed
        - Advances index with hold-last behavior
        """
        traj = self.reference_trajectory
        n_total = traj.shape[0]

        if n_total == 0:
            return np.tile(self.hold_reference, (n_pred, 1))

        start = int(self.trajectory_idx)
        end = start + n_pred

        if end <= n_total:
            window = traj[start:end, :]
        else:
            available = traj[start:, :]
            pad = np.tile(traj[-1, :], (n_pred - available.shape[0], 1))
            window = np.vstack((available, pad))
            # window = np.tile(traj[start, :], (n_pred, 1))

        # advance by one (hold last)
        if self.trajectory_idx < n_total - 1:
            self.trajectory_idx += 1
        else:
            self.trajectory_idx = n_total - 1
            self.hold_reference[:] = traj[-1, :]

        return window

    def sitl_odometry_callback(self, msg: Odometry):
        """
        Callback function to update the current state from odometry messages.
        Args:
            msg: Odometry message containing position, velocity, orientation, and angular velocity.
        """
        # Optimized: Updating existing arrays in-place
        p = msg.pose.pose.position
        self.current_state.position[:] = [p.x, p.y, p.z]

        v = msg.twist.twist.linear
        self.current_state.velocity[:] = [v.x, v.y, v.z]
        
        if not self._hold_initialized:
            self.hold_reference.fill(0.0)
            self.hold_reference[0:3] = self.current_state.position
            self.hold_reference[3:6] = self.current_state.velocity
            self._hold_initialized = True

        # Estimate acceleration from velocity change
        self._estimate_acceleration(self.current_state.velocity)

    def sitl_battery_callback(self, msg: BatteryState):
        """
        Callback function to update the battery voltage from battery status messages.
        Args:
            msg: Battery status message containing voltage information.
        """
        self.battery_voltage = msg.voltage

    def set_reference_state(self, reference_state: CrazyflieState):
        """
        Set the reference state for the Crazyflie.
        Args:
            reference_state (CrazyflieState): Desired reference state.
        """
        self.reference_state = reference_state

    def set_reference_trajectory(self, reference_trajectory: np.ndarray):
        """
        Set the reference trajectory for the Crazyflie as a NumPy matrix.
        Args:
            reference_trajectory (np.ndarray): (N, 9) array of reference states.
        """
        self.reference_trajectory = reference_trajectory
        self.trajectory_idx = 0

        if reference_trajectory is not None and reference_trajectory.size != 0:
            self.hold_reference[:] = reference_trajectory[-1, :]

    def qmt_odometry_callback(self, msg: NamedPoseArray):
        """ ROS 2 callback to process pose updates from the motion capture system. """
        for named_pose in msg.poses:
            # Optimized: check name, update in-place, and break to save cycles
            if named_pose.name == self.name:
                p = named_pose.pose.position
                lv = named_pose.velocity.linear

                self.current_state.position[:] = [p.x, p.y, p.z]
                self.current_state.velocity[:] = [lv.x, lv.y, lv.z]

                # Estimate acceleration from velocity change
                self._estimate_acceleration(self.current_state.velocity)
                break
            
        if not self._hold_initialized:
            self.hold_reference.fill(0.0)
            self.hold_reference[0:3] = self.current_state.position
            self.hold_reference[3:6] = self.current_state.velocity
            self._hold_initialized = True