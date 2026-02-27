from control_framework.control_laws.control_law import ControlLaw, ControlLawSetting
from control_framework.control_laws.models import StateSpaceModel
from control_framework.crazyflie_descriptor import CrazyflieDescriptor
import numpy as np
from scipy.linalg import solve_discrete_are


class LQRControlLawSetting(ControlLawSetting):
    """
    LQR Control Law Settings extending the base ControlLawSetting.
    Attributes:
        Q (np.ndarray): State weighting matrix.
        R (np.ndarray): Input weighting matrix.
    """
    model: StateSpaceModel
    Q: np.ndarray
    R: np.ndarray
    K: np.ndarray

    saturation_limit_neg: np.ndarray = None  # Optional saturation limit for control inputs.
    saturation_limit_pos: np.ndarray = None  # Optional saturation limit for control inputs.

    def __init__(self, model: StateSpaceModel):

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
        self.Q = np.array(Q, copy=True)
        self.R = np.array(R, copy=True)

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
            raise ValueError(
                f"Negative saturation limits shape mismatch. Expected {self.saturation_limit_neg.shape}, got {u_min.shape}."
            )
        if u_max.shape != self.saturation_limit_pos.shape:
            raise ValueError(
                f"Positive saturation limits shape mismatch. Expected {self.saturation_limit_pos.shape}, got {u_max.shape}."
            )

        self.saturation_limit_neg = np.array(u_min, copy=True)
        self.saturation_limit_pos = np.array(u_max, copy=True)


class LQRControlLaw(ControlLaw):
    """
    LQR Control Law extending the base ControlLaw.
    Attributes:
        settings (LQRControlLawSetting): Settings specific to the LQR control law.
    """
    settings: LQRControlLawSetting

    def __init__(self, settings: LQRControlLawSetting):
        super().__init__(settings)

        self.u = np.zeros((settings.model.B.shape[1],), dtype=float)

        # Pre-allocate internal buffers to avoid per-tick allocations
        self.x = np.zeros((settings.model.A.shape[0],), dtype=float)      # expected (6,)
        self.x_ref = np.zeros((settings.model.A.shape[0],), dtype=float)  # expected (6,)
        self.e = np.zeros((settings.model.A.shape[0],), dtype=float)      # expected (6,)
        self._u_raw = np.zeros((settings.model.B.shape[1],), dtype=float) # expected (3,)

        """
        Solve for the gain matrices    
        """

        A = np.asarray(self.settings.model.A, dtype=float)
        B = np.asarray(self.settings.model.B, dtype=float)

        # Ensure weights are float arrays (helps downstream BLAS paths)
        Q = np.asarray(self.settings.Q, dtype=float)
        R = np.asarray(self.settings.R, dtype=float)

        P = solve_discrete_are(A, B, Q, R)

        # K = (B^T P B + R)^-1 (B^T P A)
        BtP = B.T @ P
        self.settings.K = np.linalg.inv(BtP @ B + R) @ (BtP @ A)

        # Make sure K is a contiguous float array for fast matmul in the loop
        self.settings.K = np.asarray(self.settings.K, dtype=float)

    def compute_control_action(self, crazyflie_descriptor: CrazyflieDescriptor) -> np.ndarray:
        """
        Compute the LQR control action.
        """
        # Current state x = [p(3), v(3)]
        self.x[0:3] = crazyflie_descriptor.current_state.position
        self.x[3:6] = crazyflie_descriptor.current_state.velocity

        # Reference from trajectory stored in the descriptor (N,9)
        ref9 = crazyflie_descriptor.get_reference_point()
        self.x_ref[0:3] = ref9[0:3]
        self.x_ref[3:6] = ref9[3:6]

        # state error (in-place)
        np.subtract(self.x_ref, self.x, out=self.e)

        # optimal control (in-place)
        np.matmul(self.settings.K, self.e, out=self._u_raw)

        # Apply saturation limits (in-place)
        np.clip(
            self._u_raw,
            self.settings.saturation_limit_neg,
            self.settings.saturation_limit_pos,
            out=self.u
        )

        return self.u

    def reset(self):
        """
        Reset the internal states of the LQR controller.
        """

        self.u.fill(0)
        self.x.fill(0)
        self.x_ref.fill(0)
        self.e.fill(0)
        self._u_raw.fill(0)