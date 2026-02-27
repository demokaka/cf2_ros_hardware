from control_framework.crazyflie_descriptor import CrazyflieDescriptor
from control_framework.control_laws.control_law import ControlLaw, ControlLawSetting
from control_framework.control_laws.models import StateSpaceModel
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

        # Determine dimensions: m is inputs, p is outputs
        m = model.B.shape[1]

        self.Kp = np.zeros((m,))
        self.Ki = np.zeros((m,))
        self.Kd = np.zeros((m,))

        self.saturation_limit_pos = np.full((m,), np.inf)
        self.saturation_limit_neg = np.full((m,), -np.inf)

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
        if Kp.shape != self.Kp.shape or Ki.shape != self.Ki.shape or Kd.shape != self.Kd.shape:
            raise ValueError(f"Gain shape mismatch. Expected {self.Kp.shape}.")

        # Keep the same semantics, but store copies to avoid accidental aliasing
        self.Kp = np.array(Kp, copy=True)
        self.Ki = np.array(Ki, copy=True)
        self.Kd = np.array(Kd, copy=True)

    def set_saturation_limits(self, u_min: np.ndarray, u_max: np.ndarray):
        """
        Set the saturation limits for control inputs.
        Args:
            u_min (np.ndarray): Negative saturation limits.
            u_max (np.ndarray): Positive saturation limits.

        Raises:
            ValueError: If the shapes of the provided limits do not match the expected shapes.
        """
        if u_min.shape != self.saturation_limit_neg.shape or u_max.shape != self.saturation_limit_pos.shape:
            raise ValueError(f"Saturation limits shape mismatch. Expected {self.saturation_limit_neg.shape}.")

        # Keep the same semantics, but store copies to avoid accidental aliasing
        self.saturation_limit_neg = np.array(u_min, copy=True)
        self.saturation_limit_pos = np.array(u_max, copy=True)


class PIDControlLaw(ControlLaw):
    """
    PID Control Law extending the base ControlLaw.
    Attributes:
        settings (PIDControlLawSetting): Settings specific to the PID control law.
    """

    settings: PIDControlLawSetting

    def __init__(self, settings: PIDControlLawSetting):
        super().__init__(settings)

        self.m = settings.model.B.shape[1]
        self.p = settings.model.C.shape[0]

        # Pre-allocate internal buffers to avoid per-tick allocations
        self.integral_error = np.zeros((self.m,))
        self.previous_error = np.zeros((self.m,))
        self.u = np.zeros((self.m,))
        self.e = np.zeros((self.m,))

        # Extra buffers to avoid allocating temporaries each tick
        self._derivative_error = np.zeros((self.m,))
        self._u_raw = np.zeros((self.m,))

    def compute_control_action(self, crazyflie_descriptor: CrazyflieDescriptor) -> np.ndarray:
        """
        Compute the PID control action.
        This method should implement the PID algorithm to compute the control inputs
        based on the current state and the defined gains.

        Args:
            crazyflie_descriptor (CrazyflieDescriptor): Contains current state and reference state vector

        Returns:
            np.ndarray: The computed control input vector.
        """
        dt = self.settings.model.dt

        ref9 = crazyflie_descriptor.get_reference_point()
        p_ref = ref9[0:3]

        np.subtract(
            p_ref,
            crazyflie_descriptor.current_state.position,
            out=self.e
        )

        np.subtract(self.e, self.previous_error, out=self._derivative_error)
        self._derivative_error *= (1.0 / dt)

        self._u_raw[:] = self.settings.Kp * self.e
        self._u_raw += self.settings.Ki * self.integral_error
        self._u_raw += self.settings.Kd * self._derivative_error

        np.clip(
            self._u_raw,
            self.settings.saturation_limit_neg,
            self.settings.saturation_limit_pos,
            out=self.u
        )

        umin = self.settings.saturation_limit_neg
        umax = self.settings.saturation_limit_pos

        not_saturated_pos = self.u < umax
        not_saturated_neg = self.u > umin
        not_saturated = np.logical_and(not_saturated_pos, not_saturated_neg)

        sat_high_unwind = np.logical_and(self.u >= umax, self.e < 0.0)
        sat_low_unwind = np.logical_and(self.u <= umin, self.e > 0.0)

        can_integrate = np.logical_or(not_saturated, np.logical_or(sat_high_unwind, sat_low_unwind))

        self.integral_error += (self.e * dt) * can_integrate

        self.previous_error[:] = self.e
        return self.u

    def reset(self):
        """
        Reset integral error and previous error to zero.
        """
        self.integral_error.fill(0)
        self.previous_error.fill(0)
        self.u.fill(0)
        self.e.fill(0)
        self._derivative_error.fill(0)
        self._u_raw.fill(0)