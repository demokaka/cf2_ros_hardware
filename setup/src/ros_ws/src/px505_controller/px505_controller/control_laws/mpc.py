# mpc.py
# MPC with SAME signature everywhere:
#   compute_control_action(current_state, reference_state)
#
# current_state   = [px,py,pz,vx,vy,vz]                     shape (6,)
# reference_state =
#   - Hover:      [px,py,pz,vx,vy,vz, ax,ay,az]              shape (9,)
#   - Track:      (N+1, 9) rows of [p v a]                   shape (N+1, 9)
#
# MPC uses:
#   Xref = first 6 entries
#   Uref = last 3 entries (for k=0..N-1)
#
# Dynamics model: x_{k+1} = A x_k + B u_k, u = [ax,ay,az]
#
# Integral action (offset-free-ish):
#   xi_{k+1} = xi_k + dt * (p_k - p_ref_k)
#   cost += xi_k' Qi xi_k

from __future__ import annotations

import numpy as np
import casadi as ca

from px505_controller.control_laws.control_law import ControlLaw, ControlLawSetting
from px505_controller.control_laws.models import StateSpaceModel


class MPCControlLawSetting(ControlLawSetting):
    def __init__(self, model: StateSpaceModel, Npred: int):
        super().__init__(model)
        self.Npred = int(Npred)

        nx = model.A.shape[0]   # 6
        nu = model.B.shape[1]   # 3

        self.Q  = np.eye(nx)
        self.R  = np.eye(nu)

        self.u_min = -np.inf * np.ones(nu)
        self.u_max =  np.inf * np.ones(nu)

        self.x_min = -np.inf * np.ones(nx)
        self.x_max =  np.inf * np.ones(nx)

        self.ipopt_print_level = 0
        self.print_time = False
        self.max_iter = 100
        self.tol = 1e-6

    def set_weights(self, Q: np.ndarray, R: np.ndarray):        
        if Q.shape != (self.model.A.shape[0], self.model.A.shape[0]):
            raise ValueError(f"Q must be square with shape ({self.model.A.shape[0]}, {self.model.A.shape[0]}), got {Q.shape}")
        if R.shape != (self.model.B.shape[1], self.model.B.shape[1]):
            raise ValueError(f"R must be square with shape ({self.model.B.shape[1]}, {self.model.B.shape[1]}), got {R.shape}")
        
        self.Q = Q
        self.R = R

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
    settings: MPCControlLawSetting

    def __init__(self, settings: MPCControlLawSetting):
        super().__init__(settings)

        self.nx = settings.model.A.shape[0]
        self.nu = settings.model.B.shape[1]
        self.N  = settings.Npred
        self.dt = float(settings.model.dt)

        A = ca.DM(settings.model.A)
        B = ca.DM(settings.model.B)

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(self.nx, self.N + 1)
        U = opti.variable(self.nu, self.N)

        # Parameters
        x0   = opti.parameter(self.nx, 1)
        Xref = opti.parameter(self.nx, self.N + 1)
        Uref = opti.parameter(self.nu, self.N)

        # Weights
        Q  = ca.DM(settings.Q)
        R  = ca.DM(settings.R)

        # Initial conditions
        opti.subject_to(X[:, 0] == x0)

        # Dynamics constraints
        # Integrator: xi_{k+1} = xi_k + dt*(p_k - p_ref_k)
        # p_k = X[0:3,k]
        for k in range(self.N):
            opti.subject_to(X[:, k+1] == A @ X[:, k] + B @ U[:, k])

        # Input bounds
        umin = ca.DM(settings.u_min).reshape((self.nu, 1))
        umax = ca.DM(settings.u_max).reshape((self.nu, 1))
        for k in range(self.N):
            opti.subject_to(opti.bounded(umin, U[:, k], umax))

        # State bounds
        xmin = ca.DM(settings.x_min).reshape((self.nx, 1))
        xmax = ca.DM(settings.x_max).reshape((self.nx, 1))
        for k in range(self.N + 1):
            opti.subject_to(opti.bounded(xmin, X[:, k], xmax))

        # Cost
        cost = 0
        for k in range(self.N):
            ex = X[:, k] - Xref[:, k]
            eu = U[:, k] - Uref[:, k]
            cost += ca.mtimes([ex.T, Q, ex]) + ca.mtimes([eu.T, R, eu])

        opti.minimize(cost)

        # Solver options
        p_opts = {"print_time": bool(settings.print_time)}
        s_opts = {
            "print_level": int(settings.ipopt_print_level),
            "max_iter": int(settings.max_iter),
            "tol": float(settings.tol),
        }
        opti.solver("ipopt", p_opts, s_opts)

        # Store
        self._opti = opti
        self._X = X
        self._U = U
        self._x0 = x0
        self._Xref = Xref
        self._Uref = Uref

        self._last_X = None
        self._last_U = None
        self.u = np.zeros((self.nu,), dtype=float)

    # ----------------------------
    # Reference parsing: (9,) or (N+1,9)
    # ----------------------------
    def _format_ref9(self, reference_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          Xref_np: (N+1, nx)     [p v]
          Uref_np: (N,   nu)     [a]
          Pref_np: (N+1, 3)      [p]    (for integral dynamics)

        reference_state accepted:
          - (9,) -> repeats across horizon
          - (N+1,9) -> uses/pads/crops
        """
        ref = np.asarray(reference_state, dtype=float)

        if ref.ndim == 1:
            if ref.shape[0] != (self.nx + self.nu):
                raise ValueError(f"hover reference must be ({self.nx+self.nu},), got {ref.shape}")
            Xref_row = ref[:self.nx]
            Uref_row = ref[self.nx:self.nx+self.nu]

            Xref = np.tile(Xref_row.reshape(1, -1), (self.N + 1, 1))
            Uref = np.tile(Uref_row.reshape(1, -1), (self.N, 1))
            Pref = np.tile(Xref_row[:3].reshape(1, -1), (self.N + 1, 1))
            return Xref, Uref, Pref

        if ref.ndim == 2:
            if ref.shape[1] != (self.nx + self.nu):
                raise ValueError(f"traj reference must have {self.nx+self.nu} cols, got {ref.shape}")

            # pad/crop to N+1
            if ref.shape[0] < self.N + 1:
                pad = np.tile(ref[-1, :].reshape(1, -1), (self.N + 1 - ref.shape[0], 1))
                ref9 = np.vstack([ref, pad])
            else:
                ref9 = ref[: self.N + 1, :]

            Xref = ref9[:, :self.nx]              # (N+1,6)
            Uref_full = ref9[:, self.nx:]         # (N+1,3)
            Uref = Uref_full[: self.N, :]         # (N,3)
            Pref = Xref[:, :3]                    # (N+1,3)

            return Xref, Uref, Pref

        raise ValueError(f"reference_state must be (9,) or (N+1,9), got {ref.shape}")

    def compute_control_action(self, current_state: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
        """
        current_state: (6,) [p v]
        reference_state:
          - hover: (9,) [p v a]
          - track: (N+1,9) [p v a] over horizon
        returns u0: (3,) [ax ay az]
        """
        x = np.asarray(current_state, dtype=float).reshape(-1)
        if x.shape[0] != self.nx:
            raise ValueError(f"current_state must be ({self.nx},), got {x.shape}")

        Xref_np, Uref_np, Pref_np = self._format_ref9(reference_state)

        self._opti.set_value(self._x0, x.reshape(self.nx, 1))
        self._opti.set_value(self._Xref, Xref_np.T)   # (6, N+1)
        self._opti.set_value(self._Uref, Uref_np.T)   # (3, N)

        # Warm start
        if self._last_X is not None and self._last_U is not None:
            X_guess = np.hstack([self._last_X[:, 1:], self._last_X[:, -1:]])
            U_guess = np.hstack([self._last_U[:, 1:], self._last_U[:, -1:]])
            self._opti.set_initial(self._X, X_guess)
            self._opti.set_initial(self._U, U_guess)

        try:
            sol = self._opti.solve()
            X_sol = sol.value(self._X)
            U_sol = sol.value(self._U)
            self._last_X = X_sol
            self._last_U = U_sol
            u0 = U_sol[:, 0]

        except RuntimeError as e:
            print(f"MPC solver failed: {e}")
            if self._last_U is not None:
                u0 = self._last_U[:, 0]
            else:
                u0 = np.zeros((self.nu,), dtype=float)

            try:
                self._opti.set_initial(self._U, 0)
                self._opti.set_initial(self._X, 0)
            except Exception:
                pass

        # Saturation
        u0 = np.asarray(u0, dtype=float).reshape(-1)
        u0 = np.minimum(u0, self.settings.u_max)
        u0 = np.maximum(u0, self.settings.u_min)

        self.u = u0
        return u0

    def reset(self):
        self._last_X = None
        self._last_U = None
        self.u = np.zeros((self.nu,), dtype=float)

# Helpers to build reference_state in the "9D ref" format:
#   ref9 = [x,y,z, vx,vy,vz, ax,ay,az]

def make_hover_ref9(pos: np.ndarray, vel: np.ndarray | None = None, acc: np.ndarray | None = None) -> np.ndarray:
    pos = np.asarray(pos, float).reshape(3)
    vel = np.zeros(3) if vel is None else np.asarray(vel, float).reshape(3)
    acc = np.zeros(3) if acc is None else np.asarray(acc, float).reshape(3)
    return np.hstack([pos, vel, acc])  # (9,)

def make_traj_ref9_window(reference_trajectory, idx0: int, Npred: int) -> np.ndarray:
    """
    reference_trajectory: list of CrazyflieState with fields position, velocity, acceleration (optional)
    idx0: current trajectory index
    Returns (Npred+1, 9) padded with last sample.
    """
    idx1 = min(idx0 + Npred + 1, len(reference_trajectory))
    window = reference_trajectory[idx0:idx1]

    if len(window) == 0:
        z = np.zeros(3)
        return np.tile(np.hstack([z, z, z]).reshape(1, -1), (Npred + 1, 1))

    # pad by repeating last
    while len(window) < Npred + 1:
        window.append(window[-1])

    rows = []
    for s in window:
        p = np.asarray(s.position, float).reshape(3)
        v = np.asarray(s.velocity, float).reshape(3)
        a = getattr(s, "acceleration", None)
        a = np.zeros(3) if a is None else np.asarray(a, float).reshape(3)
        rows.append(np.hstack([p, v, a]))
    return np.vstack(rows)  # (Npred+1, 9)
