from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


# -----------------------------
# Small, explicit parse helpers
# -----------------------------

def require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a mapping/dict, got {type(value).__name__}")
    return value


def require_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{where} must be an int, got bool")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise TypeError(f"{where} must be an int, got {value!r}")


def require_float(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{where} must be a number, got bool")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{where} must be a number, got {value!r}")


def require_vector(value: Any, where: str) -> tuple[float, ...]:
    """
    Accepts list/tuple of any length >= 1. Returns tuple[float, ...].
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{where} must be a list/tuple of numbers, got {type(value).__name__}")
    if len(value) == 0:
        raise ValueError(f"{where} must not be empty")

    out: list[float] = []
    for i, x in enumerate(value):
        out.append(require_float(x, f"{where}[{i}]"))
    return tuple(out)


# -----------------------------
# Data models (objects)
# -----------------------------

@dataclass(frozen=True)
class Crazyflie:
    id: int
    url: str
    mass: float

    @staticmethod
    def from_mapping(cf_id: int, d: Mapping[str, Any], where: str) -> "Crazyflie":
        url = d.get("url")
        if url is None:
            raise ValueError(f"{where}.url is required")

        return Crazyflie(
            id=cf_id,
            url=str(url),
            mass=require_float(d.get("mass"), f"{where}.mass"),
        )


@dataclass(frozen=True)
class QuadrotorsConfig:
    crazyflies: dict[int, Crazyflie]

    @staticmethod
    def from_mapping(d: Mapping[str, Any], where: str) -> "QuadrotorsConfig":
        d = require_mapping(d, where)
        raw_cfs = require_mapping(d.get("crazyflies", {}), f"{where}.crazyflies")

        crazyflies: dict[int, Crazyflie] = {}
        for raw_id, raw_cf in raw_cfs.items():
            cf_id = require_int(raw_id, f"{where}.crazyflies key (crazyflie id)")
            cf_map = require_mapping(raw_cf, f"{where}.crazyflies[{cf_id}]")
            crazyflies[cf_id] = Crazyflie.from_mapping(cf_id, cf_map, f"{where}.crazyflies[{cf_id}]")

        return QuadrotorsConfig(crazyflies=crazyflies)


@dataclass(frozen=True)
class EnvAgentsConfig:
    quadrotors: QuadrotorsConfig

    @staticmethod
    def from_mapping(d: Mapping[str, Any], where: str) -> "EnvAgentsConfig":
        d = require_mapping(d, where)
        quad = require_mapping(d.get("quadrotors", {}), f"{where}.quadrotors")
        return EnvAgentsConfig(quadrotors=QuadrotorsConfig.from_mapping(quad, f"{where}.quadrotors"))


@dataclass(frozen=True)
class AgentsConfig:
    sitl: EnvAgentsConfig
    hitl: EnvAgentsConfig

    @staticmethod
    def from_dict(raw: Any) -> "AgentsConfig":
        d = require_mapping(raw, "agents")
        return AgentsConfig(
            sitl=EnvAgentsConfig.from_mapping(d.get("sitl", {}), "agents.sitl"),
            hitl=EnvAgentsConfig.from_mapping(d.get("hitl", {}), "agents.hitl"),
        )


# --- Controllers ---

@dataclass(frozen=True)
class PIDParameters:
    kp: tuple[float, ...]
    ki: tuple[float, ...]
    kd: tuple[float, ...]

    @staticmethod
    def from_mapping(d: Mapping[str, Any], where: str) -> "PIDParameters":
        d = require_mapping(d, where)
        kp = require_vector(d.get("Kp"), f"{where}.Kp")
        ki = require_vector(d.get("Ki"), f"{where}.Ki")
        kd = require_vector(d.get("Kd"), f"{where}.Kd")

        if not (len(kp) == len(ki) == len(kd)):
            raise ValueError(
                f"{where}: Kp/Ki/Kd must have the same length, got "
                f"{len(kp)}/{len(ki)}/{len(kd)}"
            )

        return PIDParameters(kp=kp, ki=ki, kd=kd)


@dataclass(frozen=True)
class PIDController:
    controller_id: int
    controlled_agent: int
    parameters: PIDParameters


@dataclass(frozen=True)
class EnvControllersConfig:
    pid: dict[int, PIDController]

    @staticmethod
    def from_mapping(d: Mapping[str, Any], where: str) -> "EnvControllersConfig":
        """
        Currently supports only PID as a template.
        You can extend this by adding other controller parsing branches.
        """
        raw_ctrls = require_mapping(d or {}, where)

        pid: dict[int, PIDController] = {}

        for raw_id, raw_ctrl in raw_ctrls.items():
            controller_id = require_int(raw_id, f"{where} key (controller_id)")
            ctrl = require_mapping(raw_ctrl, f"{where}[{controller_id}]")

            ctype = ctrl.get("controller_type")
            if ctype != "PID":
                raise ValueError(f"{where}[{controller_id}].controller_type must be 'PID' (got {ctype!r})")

            controlled_agent = require_int(
                ctrl.get("controlled_agent"), f"{where}[{controller_id}].controlled_agent"
            )

            params_raw = require_mapping(
                ctrl.get("parameters", {}), f"{where}[{controller_id}].parameters"
            )

            pid[controller_id] = PIDController(
                controller_id=controller_id,
                controlled_agent=controlled_agent,
                parameters=PIDParameters.from_mapping(params_raw, f"{where}[{controller_id}].parameters"),
            )

        return EnvControllersConfig(pid=pid)


@dataclass(frozen=True)
class ControllersConfig:
    sitl: EnvControllersConfig
    hitl: EnvControllersConfig

    @staticmethod
    def from_dict(raw: Any) -> "ControllersConfig":
        d = require_mapping(raw or {}, "controllers")
        return ControllersConfig(
            sitl=EnvControllersConfig.from_mapping(d.get("sitl", {}), "controllers.sitl"),
            hitl=EnvControllersConfig.from_mapping(d.get("hitl", {}), "controllers.hitl"),
        )


# -----------------------------
# Root config + validation
# -----------------------------

@dataclass(frozen=True)
class GlobalConfig:
    agents: AgentsConfig
    controllers: ControllersConfig

    @staticmethod
    def from_dict(raw: Any) -> "GlobalConfig":
        root = require_mapping(raw, "root")

        agents = AgentsConfig.from_dict(root.get("agents", {}))
        controllers = ControllersConfig.from_dict(root.get("controllers", {}))

        cfg = GlobalConfig(agents=agents, controllers=controllers)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        # Validate 1↔1 pairing per environment for PID controllers.
        self._validate_env("sitl", self.agents.sitl, self.controllers.sitl)
        self._validate_env("hitl", self.agents.hitl, self.controllers.hitl)

    @staticmethod
    def _validate_env(env: str, agents: EnvAgentsConfig, controllers: EnvControllersConfig) -> None:
        crazyflies = agents.quadrotors.crazyflies

        # Controller must reference an existing agent
        for cid, ctrl in controllers.pid.items():
            if ctrl.controlled_agent not in crazyflies:
                raise ValueError(
                    f"{env}: PID controller {cid} controls agent {ctrl.controlled_agent}, "
                    f"but agents.{env}.quadrotors.crazyflies has no such id"
                )

        # 1↔1: no two controllers control the same agent
        controlled_agents = [c.controlled_agent for c in controllers.pid.values()]
        duplicates = {a for a in controlled_agents if controlled_agents.count(a) > 1}
        if duplicates:
            dup_list = ", ".join(str(x) for x in sorted(duplicates))
            raise ValueError(f"{env}: multiple controllers assigned to the same agent id(s): {dup_list}")