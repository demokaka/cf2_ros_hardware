from px505_controller.control_laws.control_law import ControlLaw, ControlLawSetting
from px505_controller.control_laws.models import StateSpaceModel
import numpy as np
from scipy.linalg import solve_discrete_are

class LQRControlLawSetting(ControlLawSetting):
    """
    LQR Control Law Settings extending the base ControlLawSetting.
    Attributes:
        Q (np.ndarray): State weighting matrix.
        R (np.ndarray): Input weighting matrix.
    """
    model:StateSpaceModel
    Q: np.ndarray
    R: np.ndarray
    K: np.ndarray
    
    saturation_limit_neg: np.ndarray = None  # Optional saturation limit for control inputs.
    saturation_limit_pos: np.ndarray = None  # Optional saturation limit for control inputs.

    def __init__(self, model : StateSpaceModel):
        
        self.model = model
        self.Q = np.zeros((model.A.shape[1], model.A.shape[0]))
        self.R = np.zeros((model.B.shape[1], model.B.shape[1]))
        self.K = np.zeros((model.B.shape[1], model.A.shape[0]))
        
        self.saturation_limit_pos = np.full((model.B.shape[1],), np.inf)
        self.saturation_limit_neg = np.full((model.B.shape[1],), -np.inf)

    def set_weights(self, Q: np.ndarray, R: np.ndarray):
        """
        Set the weighting matrices for the LQR
        Args:
            Q  (np.ndarray): State weight matrix
            R  (np.ndarray): Input weight matrix

        Raises:
            ValueError: If the shapes of the provided gain matrices do not match the expected shapes.
        """
        if Q.shape != self.Q.shape:
            raise ValueError(f"Q shape mismatch. Expected {self.Q.shape}, got {Q.shape}.")
        if R.shape != self.R.shape:
            raise ValueError(f"R shape mismatch. Expected {self.R.shape}, got {R.shape}.")
        self.Q = Q
        self.R = R

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

class LQRControlLaw(ControlLaw):
    """
    LQR Control Law extending the base ControlLaw.
    Attributes:
        settings (LQRControlLawSetting): Settings specific to the LQR control law.
    """
    settings: LQRControlLawSetting

    def __init__(self, settings: LQRControlLawSetting):
        super().__init__(settings)

        self.u = np.zeros((settings.model.B.shape[1],))

        """
        Solve for the gain matrices    
        """

        A = self.settings.model.A
        B = self.settings.model.B

        P = solve_discrete_are(A, B, self.settings.Q, self.settings.R)
        self.settings.K = np.linalg.inv(B.T @ P @ B + self.settings.R) @ (B.T @ P @ A)
            
    def compute_control_action(self, current_state: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
        """
        Compute the LQR control action.
        """
        # state error
        error = reference_state - current_state  # expected (6,)

        # optimal control
        u = self.settings.K @ error  # -> shape (3,)

        # Apply saturation limits
        u = np.minimum(u, self.settings.saturation_limit_pos)
        u = np.maximum(u, self.settings.saturation_limit_neg)

        self.u = u
        return u

    def reset(self):
        """
        Reset the internal states of the LQR controller.
        """

        pass