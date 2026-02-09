import numpy as np
from typing import List
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from px505_controller.control_laws.control_law import ControlLaw
from motion_capture_tracking_interfaces.msg import NamedPose, NamedPoseArray # type: ignore

class CrazyflieState:
    """
    State representation for a Crazyflie drone.
    Attributes:
        position (np.ndarray): Current position of the Crazyflie.
        velocity (np.ndarray): Current velocity of the Crazyflie.
        acceleration (np.ndarray): Current acceleration of the Crazyflie.
        orientation (np.ndarray): Current orientation (quaternion) of the Crazyflie.
        angular_velocity (np.ndarray): Current angular velocity of the Crazyflie.
    """
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    
    def __init__(self, 
                 position: np.ndarray = None,
                 velocity: np.ndarray = None,
                 acceleration: np.ndarray = None,
                 orientation: np.ndarray = None,
                 angular_velocity: np.ndarray = None):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.orientation = orientation
        self.angular_velocity = angular_velocity
    
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
        
        odometry_topic (str): ROS topic for odometry data.
        battery_topic (str): ROS topic for battery data.
    """
    id: int
    uri: str
    mass: float
    current_state: CrazyflieState
    reference_state: CrazyflieState
    reference_trajectory: List[CrazyflieState]
    drone_body: str

    battery_voltage: float
    
    odometry_topic: str
    battery_topic: str
    qmt_odometry_topic: str

    def __init__(self, name, id, mass, uri):
        self.name = name
        self.id = id
        self.mass = mass
        self.uri = uri
        self.current_state = CrazyflieState()
        self.reference_state = CrazyflieState()
        self.trajectory_idx = 0
        self.reference_trajectory = []
        self.battery_voltage = 0.0

        self.odometry_topic = f"/crazyflie_{self.id}/odom"
        self.battery_topic = f"/crazyflie_{self.id}/battery_status"
        self.qmt_odometry_topic = f"/poses"
                
    def odometry_callback(self, msg: Odometry):        
        """
        Callback function to update the current state from odometry messages.
        Args:
            msg: Odometry message containing position, velocity, orientation, and angular velocity.
        """
        self.current_state.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        self.current_state.velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
        self.current_state.orientation = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ])
        
        self.current_state.angular_velocity = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])

    def battery_callback(self, msg: BatteryState):
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
        
    def set_reference_trajectory(self, reference_trajectory: List[CrazyflieState]):
        """
        Set the reference trajectory for the Crazyflie.
        Args:
            reference_trajectory (List[CrazyflieState]): Desired reference trajectory.
        """        
        self.reference_trajectory = reference_trajectory
        
    def qmt_odometry_callback(self, msg: NamedPoseArray):
        """ ROS 2 callback to process pose updates from the motion capture system. """
        for i, named_pose in enumerate(msg.poses):
            position = named_pose.pose.position
            orientation = named_pose.pose.orientation
            linear_velocity = named_pose.velocity.linear
            angular_velocity = named_pose.velocity.angular

            if named_pose.name == self.name:
                self.current_state.position = np.array([
                    position.x, position.y, position.z
                ])
                self.current_state.velocity = np.array([
                    linear_velocity.x, linear_velocity.y, linear_velocity.z
                ])
                self.current_state.orientation = np.array([
                    orientation.x, orientation.y, orientation.z, orientation.w
                ])
                self.current_state.angular_velocity = np.array([
                    angular_velocity.x, angular_velocity.y, angular_velocity.z
                ])