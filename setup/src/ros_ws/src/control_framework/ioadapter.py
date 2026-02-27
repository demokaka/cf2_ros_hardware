# control_framework/control_io.py
# Pattern: each controller gets an IOAdapter that packs exactly the state/reference it needs.
# - No per-tick allocations (preallocated buffers)
# - Works with "trajectory stored in CrazyflieDescriptor"
# - Controller loop stays identical across PID/LQR/MPC/Custom
#
# Expected CrazyflieDescriptor fields:
#   cf.current_state.position       -> (3,)
#   cf.current_state.velocity       -> (3,)
#   cf.current_state.acceleration   -> (3,)  (optional for some)
#   cf.reference_state.position     -> (3,)
#   cf.reference_state.velocity     -> (3,)
#   cf.reference_state.acceleration -> (3,)
#   cf.reference_trajectory         -> (T, 9) [p(3) v(3) a(3)]
#   cf.trajectory_idx               -> int
#
# Recommended: pad the stored trajectory ONCE (at load time) with (Npred+1) rows of the last sample,
# so slicing never needs allocation.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Optional
import numpy as np


# -----------------------------
# Interfaces
# -----------------------------

class IOAdapter(Protocol):
    """Packs state and reference for a control law. Should be allocation-free per tick."""
    def get_x(self, cf: Any) -> np.ndarray: ...
    def get_ref(self, cf: Any) -> Any: ...


# -----------------------------
# Concrete adapters
# -----------------------------

class PIDIO:
    """
    PID expects:
      x   = position (3,)
      ref = position_ref (3,)
    """
    __slots__ = ("_x", "_r")

    def __init__(self):
        self._x = np.zeros(3, dtype=float)
        self._r = np.zeros(3, dtype=float)

    def get_x(self, cf: Any) -> np.ndarray:
        self._x[:] = cf.current_state.position
        return self._x

    def get_ref(self, cf: Any) -> np.ndarray:
        self._r[:] = cf.reference_state.position
        return self._r


class LQRIO:
    """
    LQR expects:
      x   = [p, v] (6,)
      ref = [p_ref, v_ref] (6,)
    """
    __slots__ = ("_x", "_r")

    def __init__(self):
        self._x = np.zeros(6, dtype=float)
        self._r = np.zeros(6, dtype=float)

    def get_x(self, cf: Any) -> np.ndarray:
        self._x[0:3] = cf.current_state.position
        self._x[3:6] = cf.current_state.velocity
        return self._x

    def get_ref(self, cf: Any) -> np.ndarray:
        self._r[0:3] = cf.reference_state.position
        self._r[3:6] = cf.reference_state.velocity
        return self._r


class MPCIO:
    """
    MPC expects:
      x   = [p, v] (6,)
      ref = (Npred+1, 9) trajectory window of [p v a]
    """
    __slots__ = ("_x", "Npred")

    def __init__(self, Npred: int):
        self._x = np.zeros(6, dtype=float)
        self.Npred = int(Npred)

    def get_x(self, cf: Any) -> np.ndarray:
        self._x[0:3] = cf.current_state.position
        self._x[3:6] = cf.current_state.velocity
        return self._x

    def get_ref(self, cf: Any) -> np.ndarray:
        # Fast path: view slice (no alloc) if trajectory is padded once at load time.
        i = int(cf.trajectory_idx)
        j = i + self.Npred + 1
        return cf.reference_trajectory[i:j, :]


class FullStateRef9IO:
    """
    Optional "unified 9D ref" adapter:
      x   = [p, v] (6,)
      ref = (9,) [p_ref, v_ref, a_ref]
    Useful if you want one adapter that feeds both LQR/PID-style laws that accept ref vectors.
    """
    __slots__ = ("_x", "_ref9")

    def __init__(self):
        self._x = np.zeros(6, dtype=float)
        self._ref9 = np.zeros(9, dtype=float)

    def get_x(self, cf: Any) -> np.ndarray:
        self._x[0:3] = cf.current_state.position
        self._x[3:6] = cf.current_state.velocity
        return self._x

    def get_ref(self, cf: Any) -> np.ndarray:
        self._ref9[0:3] = cf.reference_state.position
        self._ref9[3:6] = cf.reference_state.velocity
        self._ref9[6:9] = cf.reference_state.acceleration
        return self._ref9


# -----------------------------
# Utilities
# -----------------------------

def pad_trajectory_once(traj: np.ndarray, pad_rows: int) -> np.ndarray:
    """
    Returns a trajectory padded by repeating its last row `pad_rows` times.
    This lets MPC windows be pure slices (no allocations) for the whole run.
    """
    traj = np.asarray(traj, dtype=float)
    if pad_rows <= 0:
        return traj
    if traj.size == 0:
        return traj
    last = traj[-1:, :]
    pad = np.repeat(last, pad_rows, axis=0)
    return np.vstack([traj, pad])


# -----------------------------
# Factory
# -----------------------------

@dataclass(frozen=True)
class AdapterSpec:
    controller_type: str
    horizon: Optional[int] = None  # for MPC


def make_io_adapter(spec: AdapterSpec) -> IOAdapter:
    """
    Map a controller type to the correct IO adapter.
    Extend this with your CustomControlLaw adapters.
    """
    ct = spec.controller_type.upper()

    if ct == "PID":
        return PIDIO()

    if ct == "LQR":
        return LQRIO()

    if ct == "MPC":
        if spec.horizon is None:
            raise ValueError("MPC requires horizon in AdapterSpec")
        return MPCIO(Npred=spec.horizon)

    # For "Custom", you have two options:
    #  1) default to something generic
    #  2) load an adapter class dynamically via import path (not included here)
    if ct == "CUSTOM":
        # default: give custom controller full ref9 + state (you can replace this)
        return FullStateRef9IO()

    raise ValueError(f"Unknown controller_type '{spec.controller_type}'")


# -----------------------------
# Example usage in your ControllerNode loop (pseudo)
# -----------------------------
#
#   # at init:
#   law = ...  # PID/LQR/MPC instance
#   spec = AdapterSpec(controller_type=controller_config.controller_type,
#                      horizon=getattr(controller_config.parameters, "horizon", None))
#   io = make_io_adapter(spec)
#
#   # if MPC and you store trajectory in descriptor:
#   if spec.controller_type.upper() == "MPC":
#       cf.reference_trajectory = pad_trajectory_once(cf.reference_trajectory, pad_rows=spec.horizon + 1)
#
#   # in control loop (same for all):
#   x = io.get_x(cf)          # np.ndarray
#   ref = io.get_ref(cf)      # np.ndarray or other
#   u = law.compute_control_action(x, ref)