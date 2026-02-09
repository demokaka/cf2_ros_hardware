from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PIDParameters:
    kp: tuple[float, ...]
    ki: tuple[float, ...]
    kd: tuple[float, ...]
    dt: int  # ms
    u_min: tuple[float, ...] = (float("-inf"),)  # optional saturation limits
    u_max: tuple[float, ...] = (float("inf"),)


@dataclass(frozen=True)
class PIDController:
    name: str                 # YAML key (controller instance name)
    controller_type: str      # "PID"
    controlled_agent_id: int  # numeric id
    parameters: PIDParameters


# Add new controller types as new dataclasses later, e.g.:
# @dataclass(frozen=True)
# class LQRController: ...

@dataclass(frozen=True)
class LQRParameters:
    Q: list[float] # state cost diagonal
    R: list[float] # control cost diagonal
    dt: int        # ms
    u_min: list[float] = None  # optional saturation limits
    u_max: list[float] = None  # optional saturation limits

@dataclass(frozen=True)
class LQRController:
    name: str
    controller_type: str  # "LQR"
    controlled_agent_id: int
    parameters: LQRParameters

# class MPCController: ...

@dataclass(frozen=True)
class MPCParameters:
    Q: list[float] # state cost diagonal
    R: list[float] # control cost diagonal
    dt: int        # ms
    horizon: int   # number of steps in the prediction horizon
    u_min: list[float] = None  # optional saturation limits
    u_max: list[float] = None  # optional saturation limits
    x_min: list[float] = None  # optional state constraints
    x_max: list[float] = None  # optional state constraints

@dataclass(frozen=True)
class MPCController:
    name: str
    controller_type: str  # "MPC"
    controlled_agent_id: int
    parameters: MPCParameters

Controller = Union[PIDController, LQRController, MPCController]  # extend union as you add types


@dataclass(frozen=True)
class EnvControllersConfig:
    controllers: dict[str, Controller]  # by controller instance name


@dataclass(frozen=True)
class ControllersConfig:
    sitl: EnvControllersConfig
    hitl: EnvControllersConfig
