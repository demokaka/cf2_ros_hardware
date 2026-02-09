from px505_controller.control_laws.control_law import ControlLaw, ControlLawSetting
from px505_controller.control_laws.models import StateSpaceModel
from px505_controller.control_laws.filter import DigitalFilter
import numpy as np

class PIDControlLawSetting(ControlLawSetting):
    """
    PID Control Law Settings extending the base ControlLawSetting.
    Attributes:
        Kp (np.ndarray): Proportional gain vector.
        Ki (np.ndarray): Integral gain vector.
        Kd (np.ndarray): Derivative gain vector.
    """
    Kp: np.ndarray
    Ki: np.ndarray
    Kd: np.ndarray    
    
    saturation_limit_neg: np.ndarray = None  # Optional saturation limit for control inputs.
    saturation_limit_pos: np.ndarray = None  # Optional saturation limit for control inputs.
    
    def __init__(self, model: StateSpaceModel):
        super().__init__(model)

        self.Kp = np.zeros((model.B.shape[1],))
        self.Ki = np.zeros((model.B.shape[1],))
        self.Kd = np.zeros((model.B.shape[1],))
        
        self.saturation_limit_pos = np.full((model.B.shape[1],), np.inf)
        self.saturation_limit_neg = np.full((model.B.shape[1],), -np.inf)
        
    def set_gains(self, Kp: np.ndarray, Ki: np.ndarray, Kd: np.ndarray):
        """
        Set the PID gains.
        Args:
            Kp (np.ndarray): Proportional gain vector.
            Ki (np.ndarray): Integral gain vector.
            Kd (np.ndarray): Derivative gain vector.
            
        Raises:
            ValueError: If the shapes of the provided gain vectors do not match the expected shapes.
        """
        
        if Kp.shape != self.Kp.shape:
            raise ValueError(f"Kp shape mismatch. Expected {self.Kp.shape}, got {Kp.shape}.")
        if Ki.shape != self.Ki.shape:
            raise ValueError(f"Ki shape mismatch. Expected {self.Ki.shape}, got {Ki.shape}.")
        if Kd.shape != self.Kd.shape:
            raise ValueError(f"Kd shape mismatch. Expected {self.Kd.shape}, got {Kd.shape}.")
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd    

    def set_saturation_limits(self, u_min: np.ndarray, u_max: np.ndarray):
        """
        Set the saturation limits for control inputs.
        Args:
            u_min (np.ndarray): Negative saturation limits.
            u_max (np.ndarray): Positive saturation limits.
            
        Raises:
            ValueError: If the shapes of the provided limits do not match the expected shapes.
        """
        
        if u_min.shape != self.saturation_limit_neg.shape:
            raise ValueError(f"Negative saturation limits shape mismatch. Expected {self.saturation_limit_neg.shape}, got {u_min.shape}.")
        if u_max.shape != self.saturation_limit_pos.shape:
            raise ValueError(f"Positive saturation limits shape mismatch. Expected {self.saturation_limit_pos.shape}, got {u_max.shape}.")
        
        self.saturation_limit_neg = u_min
        self.saturation_limit_pos = u_max

class PIDControlLaw(ControlLaw):
    """
    PID Control Law extending the base ControlLaw.
    Attributes:
        settings (PIDControlLawSetting): Settings specific to the PID control law.
    """
    
    settings: PIDControlLawSetting

    def __init__(self, settings: PIDControlLawSetting):
        super().__init__(settings)

        m = settings.model.B.shape[1]
        self.integral_error = np.zeros((m,))
        self.previous_error = np.zeros((m,))
        self.u = np.zeros((m,))

    def compute_control_action(self, current_state: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
        """
        Compute the PID control action.
        This method should implement the PID algorithm to compute the control inputs
        based on the current state and the defined gains.

        Args:
            current_state (np.ndarray): The current state vector.
            reference_state (np.ndarray): The desired reference state vector.
            
        Returns:
            np.ndarray: The computed control input vector.
        """
        m = self.settings.model.B.shape[1]

        current_state = np.asarray(current_state, float).reshape(-1)
        reference_state = np.asarray(reference_state, float).reshape(-1)

        if current_state.shape != (m,) or reference_state.shape != (m,):
            raise ValueError(
                f"PID expects current/reference shape {(m,)}, got "
                f"{current_state.shape} and {reference_state.shape}"
            )

        error = reference_state - current_state
        derivative_error = (error - self.previous_error) / self.settings.model.dt

        # Anti-windup: only integrate if we're not saturated
        if not (np.any(self.u >= self.settings.saturation_limit_pos) or np.any(self.u <= self.settings.saturation_limit_neg)):
            self.integral_error += error * self.settings.model.dt

        # Elementwise PID (vector gains)
        u = (
            self.settings.Kp * error
            + self.settings.Ki * self.integral_error
            + self.settings.Kd * derivative_error
        )

        # Apply saturation
        u = np.minimum(u, self.settings.saturation_limit_pos)
        u = np.maximum(u, self.settings.saturation_limit_neg)

        self.u = u
        self.previous_error = error
        return u

    def reset(self):
        """
        Reset integral error and previous error to zero.
        """
        
        m = self.settings.model.B.shape[1]
        self.integral_error = np.zeros((m,))
        self.previous_error = np.zeros((m,))
        self.u = np.zeros((m,))
