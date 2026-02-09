from __future__ import annotations
from typing import Any, Callable

from px505_controller.config.models.controllers import (
    LQRParameters,
    LQRController,
    MPCController,
    MPCParameters,
    PIDParameters,
    PIDController,
    Controller,
    EnvControllersConfig,
    ControllersConfig,
)

from px505_controller.config.parsing.helpers import require_mapping, require_int, require_vector


def parse_pid_parameters(raw: Any, where: str) -> PIDParameters:
    d = require_mapping(raw or {}, where)
    kp = require_vector(d.get("Kp"), f"{where}.Kp")
    ki = require_vector(d.get("Ki"), f"{where}.Ki")
    kd = require_vector(d.get("Kd"), f"{where}.Kd")
    dt = require_int(d.get("dt"), f"{where}.dt")

    u_min = require_vector(d.get("u_min", []), f"{where}.u_min") if "u_min" in d else (float("-inf"),)
    u_max = require_vector(d.get("u_max", []), f"{where}.u_max") if "u_max" in d else (float("inf"),)

    if dt <= 0:
        raise ValueError(f"{where}.dt must be positive non-zero, got {dt}")

    if not (len(kp) == len(ki) == len(kd)):
        raise ValueError(
            f"{where}: Kp/Ki/Kd must have the same length, got {len(kp)}/{len(ki)}/{len(kd)}"
        )

    if len(u_min) != len(u_max):
        raise ValueError(
            f"{where}: u_min and u_max must have the same length, got {len(u_min)}/{len(u_max)}"
        )
    if any(um >= ux for um, ux in zip(u_min, u_max)):
        raise ValueError(
            f"{where}: each element of u_min must be less than corresponding element of u_max, got {u_min} and {u_max}"
        )
    
    return PIDParameters(kp=kp, ki=ki, kd=kd, dt=dt, u_min=u_min, u_max=u_max)


def parse_pid_controller(name: str, raw: dict[str, Any], where: str) -> PIDController:
    controlled_agent_id = require_int(raw.get("controlled_agent_id"), f"{where}.controlled_agent_id")
    params = parse_pid_parameters(raw.get("parameters", {}), f"{where}.parameters")
    return PIDController(
        name=name,
        controller_type="PID",
        controlled_agent_id=controlled_agent_id,
        parameters=params,
    )

def parse_lqr_parameters(raw: Any, where: str) -> LQRParameters:
    d = require_mapping(raw or {}, where)
    Q = require_vector(d.get("Q"), f"{where}.Q")
    R = require_vector(d.get("R"), f"{where}.R")
    dt = require_int(d.get("dt"), f"{where}.dt")

    u_min = require_vector(d.get("u_min", []), f"{where}.u_min") if "u_min" in d else None
    u_max = require_vector(d.get("u_max", []), f"{where}.u_max") if "u_max" in d else None

    if dt <= 0:
        raise ValueError(f"{where}.dt must be positive non-zero, got {dt}")

    if len(R) != len(u_min) or len(R) != len(u_max):
        raise ValueError(
            f"{where}: R length must match u_min/u_max length, got {len(R)}/{len(u_min)}/{len(u_max)}"
        )
    if any(um >= ux for um, ux in zip(u_min, u_max)):
        raise ValueError(
            f"{where}: each element of u_min must be less than corresponding element of u_max, got {u_min} and {u_max}"
        )

    return LQRParameters(Q=Q, R=R, dt=dt, u_min=u_min, u_max=u_max)

def parse_lqr_controller(name: str, raw: dict[str, Any], where: str) -> LQRController:
    controlled_agent_id = require_int(raw.get("controlled_agent_id"), f"{where}.controlled_agent_id")
    params = parse_lqr_parameters(raw.get("parameters", {}), f"{where}.parameters")
    return LQRController(
        name=name,
        controller_type="LQR",
        controlled_agent_id=controlled_agent_id,
        parameters=params,
    )

def parse_mpc_parameters(raw: Any, where: str) -> MPCParameters:
    d = require_mapping(raw or {}, where)
    Q = require_vector(d.get("Q"), f"{where}.Q")
    R = require_vector(d.get("R"), f"{where}.R")
    dt = require_int(d.get("dt"), f"{where}.dt")
    horizon = require_int(d.get("horizon"), f"{where}.horizon")

    u_min = require_vector(d.get("u_min", []), f"{where}.u_min") if "u_min" in d else None
    u_max = require_vector(d.get("u_max", []), f"{where}.u_max") if "u_max" in d else None
    x_min = require_vector(d.get("x_min", []), f"{where}.x_min") if "x_min" in d else None
    x_max = require_vector(d.get("x_max", []), f"{where}.x_max") if "x_max" in d else None

    if dt <= 0:
        raise ValueError(f"{where}.dt must be positive non-zero, got {dt}")
    if horizon <= 0:
        raise ValueError(f"{where}.horizon must be positive non-zero, got {horizon}")
    
    if len(R) != len(u_min) or len(R) != len(u_max):
        raise ValueError(
            f"{where}: R length must match u_min/u_max length, got {len(R)}/{len(u_min)}/{len(u_max)}"
        )
    if any(um >= ux for um, ux in zip(u_min, u_max)):
        raise ValueError(
            f"{where}: each element of u_min must be less than corresponding element of u_max, got {u_min} and {u_max}"
        )
    
    if len(Q) != len(x_min) or len(Q) != len(x_max):
        raise ValueError(
            f"{where}: Q length must match x_min/x_max length, got {len(Q)}/{len(x_min)}/{len(x_max)}"
        )
    if any(xm >= xx for xm, xx in zip(x_min, x_max)):
        raise ValueError(
            f"{where}: each element of x_min must be less than corresponding element of x_max, got {x_min} and {x_max}"
        )
        
    return MPCParameters(Q=Q, R=R, dt=dt, horizon=horizon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

def parse_mpc_controller(name: str, raw: dict[str, Any], where: str) -> MPCController:
    controlled_agent_id = require_int(raw.get("controlled_agent_id"), f"{where}.controlled_agent_id")
    params = parse_mpc_parameters(raw.get("parameters", {}), f"{where}.parameters")
    return MPCController(
        name=name,
        controller_type="MPC",
        controlled_agent_id=controlled_agent_id,
        parameters=params,
    )

# Controller type dispatch table (extend this as you add controllers)
PARSERS: dict[str, Callable[[str, dict[str, Any], str], Controller]] = {
    "PID": parse_pid_controller,
    "LQR": parse_lqr_controller,
    "MPC": parse_mpc_controller,
}


def parse_env_controllers(raw: Any, where: str) -> EnvControllersConfig:
    raw_ctrls = require_mapping(raw or {}, where)
    controllers: dict[str, Controller] = {}

    for name, raw_ctrl in raw_ctrls.items():
        name = str(name)
        ctrl = require_mapping(raw_ctrl, f"{where}[{name}]")

        ctype = ctrl.get("controller_type")
        if not isinstance(ctype, str):
            raise TypeError(f"{where}[{name}].controller_type must be a string, got {ctype!r}")

        parser = PARSERS.get(ctype)
        if parser is None:
            supported = ", ".join(sorted(PARSERS.keys()))
            raise ValueError(f"{where}[{name}]: unsupported controller_type {ctype!r}. Supported: {supported}")

        controllers[name] = parser(name, dict(ctrl), f"{where}[{name}]")

    return EnvControllersConfig(controllers=controllers)


def parse_controllers(raw: Any) -> ControllersConfig:
    d = require_mapping(raw or {}, "controllers")
    return ControllersConfig(
        sitl=parse_env_controllers(d.get("sitl", {}), "controllers.sitl"),
        hitl=parse_env_controllers(d.get("hitl", {}), "controllers.hitl"),
    )
