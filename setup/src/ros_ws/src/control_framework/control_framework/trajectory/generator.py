import numpy as np
import casadi as ca
from typing import List, Optional
from control_framework.control_laws.models import TripleIntegrator3DOF

class TrajectoryGenerator:
    """
    Offline (batch) waypoint trajectory generator using CasADi.
    Minimum jerk formulation
    """

    def __init__(
        self,
        dt: float,

        R: np.ndarray = None,

        x_min: np.ndarray = None,
        x_max: np.ndarray = None,

        end_stop: bool = True,
    ):
        self.dt = float(dt)

        self.model = TripleIntegrator3DOF.from_dt(self.dt)
        self.A = np.asarray(self.model.A, dtype=float)
        self.B = np.asarray(self.model.B, dtype=float)
        self.nx = int(self.A.shape[0])  # 9
        self.nu = int(self.B.shape[1])  # 3

        if R is None:
            # Penalize Jerk (u) - R here acts on the control effort
            self.R = np.eye(self.nu) 
        else:
            self.R = np.asarray(R, dtype=float)

        # Bounds
        self.x_min = np.array(x_min, dtype=float).reshape(self.nx,) if x_min is not None else np.full((self.nx,), -np.inf)
        self.x_max = np.array(x_max, dtype=float).reshape(self.nx,) if x_max is not None else np.full((self.nx,), +np.inf)
        # self.u_min = np.array(u_min, dtype=float).reshape(self.nu,) if u_min is not None else np.full((self.nu,), -np.inf)
        # self.u_max = np.array(u_max, dtype=float).reshape(self.nu,) if u_max is not None else np.full((self.nu,), +np.inf)

        if np.any(self.x_min > self.x_max):
            raise ValueError("Invalid state bounds: some x_min > x_max")
        # if np.any(self.u_min > self.u_max):
        #     raise ValueError("Invalid input bounds: some u_min > u_max")

        self.end_stop = bool(end_stop)

        # Each entry: (wp_xyz, dt_from_prev)
        # - First waypoint dt_from_prev is ignored (None)
        self._wps: List[tuple[np.ndarray, Optional[float]]] = []

    def clear_waypoints(self) -> None:
        self._wps.clear()

    def add_waypoint(self, waypoint_xyz, dt: float = None) -> None:
        wp = np.asarray(waypoint_xyz, dtype=float).reshape((3,))

        if len(self._wps) == 0:
            # first waypoint (init knot)
            self._wps.append((wp, 0))
            return

        if dt is None:
            raise ValueError("Timed-only mode: wp requires dt (seconds). Example: wp ... x y z 2.5")
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("Waypoint dt must be > 0 (seconds)")

        self._wps.append((wp, dt))

    def prepare_lin_xref(self) -> np.ndarray:
        """
        Creates a linear interpolation of positions for the initial guess.
        Returns (N+1, 3)
        """
        if len(self._wps) < 2:
            return np.array([])

        total_steps = sum(int(np.round(w[1] / self.dt)) for w in self._wps[1:])
        xref = np.zeros((total_steps + 1, 3))
        
        curr_step = 0
        xref[0] = self._wps[0][0]
        
        for i in range(1, len(self._wps)):
            start_pos = self._wps[i-1][0]
            end_pos = self._wps[i][0]
            steps = int(np.round(self._wps[i][1] / self.dt))
            
            for s in range(1, steps + 1):
                frac = s / steps
                xref[curr_step + s] = start_pos + frac * (end_pos - start_pos)
            curr_step += steps
            
        return xref

    def generate(self, x0: np.ndarray = None) -> np.ndarray:
        """
        Solves the optimization problem using CasADi.
        Returns traj (N+1, 9): [pos(3), vel(3), acc(3)].
        """
        if len(self._wps) < 2:
            return np.empty((0, self.nx))
        
        wp_times = np.cumsum([w[1] for w in self._wps])
        wp_indices = [int(np.round(t / self.dt)) for t in wp_times]
        N = wp_indices[-1] # Total horizon

        if x0 is None:
            x0 = np.zeros(self.nx)
            x0[:3] = self._wps[0][0]

        opti = ca.Opti()
        X = opti.variable(self.nx, N + 1)
        U = opti.variable(self.nu, N)

        opti.subject_to(X[:, 0] == x0) # Initial state
        
        for k in range(N):
            opti.subject_to(X[:, k+1] == self.A @ X[:, k] + self.B @ U[:, k])
            
            opti.subject_to(opti.bounded(self.x_min, X[:, k+1], self.x_max))
            # opti.subject_to(opti.bounded(self.u_min, U[:, k], self.u_max))

        for i, idx in enumerate(wp_indices):
            opti.subject_to(X[:3, idx] == self._wps[i][0])

        if self.end_stop:
            opti.subject_to(X[3:, N] == 0)

        obj = 0
        for k in range(N):
            obj += ca.mtimes([U[:, k].T, self.R, U[:, k]])
        opti.minimize(obj)

        xref = self.prepare_lin_xref()
        opti.set_initial(X[:3, :], xref.T)
        
        opts = {
            "ipopt.print_level": 0, 
            "print_time": 0, 
            "ipopt.sb": "yes"
        }
        opti.solver('ipopt', opts)
        
        try:
            sol = opti.solve()
            return sol.value(X).T
        except RuntimeError:
            print("Optimization failed! Returning zeros.")
            return np.zeros((N + 1, self.nx))