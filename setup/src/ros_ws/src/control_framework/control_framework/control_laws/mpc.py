from __future__ import annotations

import numpy as np
import casadi as ca

from control_framework.crazyflie_descriptor import CrazyflieDescriptor
from control_framework.control_laws.control_law import ControlLaw, ControlLawSetting
from control_framework.control_laws.models import StateSpaceModel


class MPCControlLawSetting(ControlLawSetting):
    """
    MPC Control Law Settings extending the base ControlLawSetting.

    Attributes:
        model (StateSpaceModel): Discrete-time prediction model.
        Npred (int): Prediction horizon length.

        Q (np.ndarray): State weighting matrix.
        R (np.ndarray): Input weighting matrix.

        u_min (np.ndarray): Lower bounds on control input.
        u_max (np.ndarray): Upper bounds on control input.

        x_min (np.ndarray): Lower bounds on state.
        x_max (np.ndarray): Upper bounds on state.

        ipopt_print_level (int): IPOPT verbosity level.
        print_time (bool): Whether IPOPT prints timing.
        max_iter (int): Maximum solver iterations.
        tol (float): Convergence tolerance.
    """
    def __init__(self, model: StateSpaceModel, Npred: int):
        super().__init__(model)
        self.Npred = int(Npred)

        nx = model.A.shape[0]   # 6
        nu = model.B.shape[1]   # 3

        self.Q  = np.eye(nx)
        self.Qf = np.eye(nx)
        self.R  = np.eye(nu)

        self.u_min = -np.inf * np.ones(nu)
        self.u_max =  np.inf * np.ones(nu)

        self.x_min = -np.inf * np.ones(nx)
        self.x_max =  np.inf * np.ones(nx)

        self.ipopt_print_level = 0
        self.print_time = False
        self.max_iter = 100
        self.tol = 1e-6

    def set_weights(self, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray):        
        if Q.shape != (self.model.A.shape[0], self.model.A.shape[0]):
            raise ValueError(f"Q must be square with shape ({self.model.A.shape[0]}, {self.model.A.shape[0]}), got {Q.shape}")
        if Qf.shape != (self.model.A.shape[0], self.model.A.shape[0]):
            raise ValueError(f"Qf must be square with shape ({self.model.A.shape[0]}, {self.model.A.shape[0]}), got {Qf.shape}")
        if R.shape != (self.model.B.shape[1], self.model.B.shape[1]):
            raise ValueError(f"R must be square with shape ({self.model.B.shape[1]}, {self.model.B.shape[1]}), got {R.shape}")
        
        self.Q  = Q
        self.R  = R
        self.Qf = Qf

    def set_input_constraints(self, u_min: np.ndarray, u_max: np.ndarray):
        if u_min.shape != (self.model.B.shape[1],):
            raise ValueError(f"u_min must have shape ({self.model.B.shape[1]},), got {u_min.shape}")
        if u_max.shape != (self.model.B.shape[1],):
            raise ValueError(f"u_max must have shape ({self.model.B.shape[1]},), got {u_max.shape}")
        self.u_min = u_min
        self.u_max = u_max

    def set_state_constraints(self, x_min: np.ndarray, x_max: np.ndarray):
        if x_min.shape != (self.model.A.shape[0],):
            raise ValueError(f"x_min must have shape ({self.model.A.shape[0]},), got {x_min.shape}")
        if x_max.shape != (self.model.A.shape[0],):
            raise ValueError(f"x_max must have shape ({self.model.A.shape[0]},), got {x_max.shape}")
        self.x_min = x_min
        self.x_max = x_max


class MPCControlLaw(ControlLaw):
    """
    MPC Control Law extending the base ControlLaw.

    At every control iteration it solves:

        minimize   Σ (x_k - x_ref_k)' Q (x_k - x_ref_k)
                 +  Σ (u_k - u_ref_k)' R (u_k - u_ref_k)

        subject to:
            x_{k+1} = A x_k + B u_k
            x_min <= x_k <= x_max
            u_min <= u_k <= u_max

    Reference handling:
        - Uses CrazyflieDescriptor.get_mpc_window(N+1)
        - Descriptor manages trajectory indexing and hold-last behavior
    """

    settings: MPCControlLawSetting

    def __init__(self, settings: MPCControlLawSetting):
        super().__init__(settings)

        self.nx = settings.model.A.shape[0]
        self.nu = settings.model.B.shape[1]
        self.N  = settings.Npred

        A = ca.DM(settings.model.A)
        B = ca.DM(settings.model.B)

        opti = ca.Opti()

        X = opti.variable(self.nx, self.N + 1)
        U = opti.variable(self.nu, self.N)

        x0   = opti.parameter(self.nx, 1)
        Xref = opti.parameter(self.nx, self.N + 1)
        Uref = opti.parameter(self.nu, self.N)

        Q = ca.DM(settings.Q)
        R = ca.DM(settings.R)
        Qf = ca.DM(settings.Qf)

        opti.subject_to(X[:, 0] == x0)

        for k in range(self.N):
            opti.subject_to(X[:, k+1] == A @ X[:, k] + B @ U[:, k])

        umin = ca.DM(settings.u_min).reshape((self.nu, 1))
        umax = ca.DM(settings.u_max).reshape((self.nu, 1))
        for k in range(self.N):
            opti.subject_to(opti.bounded(umin, U[:, k], umax))

        xmin = ca.DM(settings.x_min).reshape((self.nx, 1))
        xmax = ca.DM(settings.x_max).reshape((self.nx, 1))
        for k in range(self.N + 1):
            opti.subject_to(opti.bounded(xmin, X[:, k], xmax))

        cost = 0
        for k in range(self.N):
            ex = X[:, k] - Xref[:, k]
            eu = U[:, k] - Uref[:, k]
            cost += ca.mtimes([ex.T, Q, ex]) + ca.mtimes([eu.T, R, eu])
        cost += ca.mtimes([(X[:, self.N] - Xref[:, self.N]).T, Qf, (X[:, self.N] - Xref[:, self.N])])

        opti.minimize(cost)

        opti.solver(
            "ipopt",
            {"print_time": settings.print_time},
            {
                "print_level": settings.ipopt_print_level,
                "max_iter": settings.max_iter,
                "tol": settings.tol,
            },
        )

        self._opti = opti
        self._X = X
        self._U = U
        self._x0 = x0
        self._Xref = Xref
        self._Uref = Uref

        self.u = np.zeros((self.nu,), dtype=float)

        self._x0_buf = np.zeros((self.nx, 1))
        self._Xref_buf = np.zeros((self.nx, self.N + 1))
        self._Uref_buf = np.zeros((self.nu, self.N))

    def compute_control_action(self, crazyflie_descriptor: CrazyflieDescriptor) -> np.ndarray:
        """
        Compute the MPC control action.
        """

        self._x0_buf[0:3, 0] = crazyflie_descriptor.current_state.position
        self._x0_buf[3:6, 0] = crazyflie_descriptor.current_state.velocity

        if crazyflie_descriptor.reference_trajectory.size == 0:
            ref9 = np.tile(crazyflie_descriptor.hold_reference, (self.N + 1, 1))
        else:
            ref9 = crazyflie_descriptor.get_mpc_window(self.N + 1)

        self._Xref_buf[:] = ref9[:, :self.nx].T
        self._Uref_buf[:] = ref9[:self.N, self.nx:].T

        self._opti.set_value(self._x0, self._x0_buf)
        self._opti.set_value(self._Xref, self._Xref_buf)
        self._opti.set_value(self._Uref, self._Uref_buf)

        sol = self._opti.solve()
        _U = np.asarray(sol.value(self._U))
        self.u = _U[:, 0]

        return self.u

    def reset(self):
        """
        Reset the internal states of the MPC controller.
        """
        self.u.fill(0.0)

        try:
            self._opti.set_initial(self._X, 0)
            self._opti.set_initial(self._U, 0)
        except Exception:
            pass