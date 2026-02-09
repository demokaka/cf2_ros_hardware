from __future__ import annotations
from typing import Any

from px505_controller.config.models.agents import Crazyflie, QuadrotorsConfig, EnvAgentsConfig, AgentsConfig
from px505_controller.config.parsing.helpers import require_mapping, require_int, require_float


def parse_crazyflies(raw: Any, where: str) -> QuadrotorsConfig:
    raw_cfs = require_mapping(raw or {}, where)

    crazyflies: dict[str, Crazyflie] = {}
    id_to_name: dict[int, str] = {}

    for name, raw_cf in raw_cfs.items():
        name = str(name)
        cf_map = require_mapping(raw_cf, f"{where}[{name}]")

        cf_id = require_int(cf_map.get("id"), f"{where}[{name}].id")
        if cf_id in id_to_name:
            raise ValueError(
                f"{where}: duplicate crazyflie id {cf_id} (used by {id_to_name[cf_id]!r} and {name!r})"
            )
        id_to_name[cf_id] = name

        uri = cf_map.get("uri")
        if uri is None:
            raise ValueError(f"{where}[{name}].uri is required")

        crazyflies[name] = Crazyflie(
            name=name,
            id=cf_id,
            uri=str(uri),
            mass=require_float(cf_map.get("mass"), f"{where}[{name}].mass"),
        )

    return QuadrotorsConfig(crazyflies=crazyflies, id_to_name=id_to_name)


def parse_env_agents(raw: Any, where: str) -> EnvAgentsConfig:
    d = require_mapping(raw or {}, where)
    quad = require_mapping(d.get("quadrotors", {}), f"{where}.quadrotors")
    quad_cfg = parse_crazyflies(quad.get("crazyflies", {}), f"{where}.quadrotors.crazyflies")
    return EnvAgentsConfig(quadrotors=quad_cfg)


def parse_agents(raw: Any) -> AgentsConfig:
    d = require_mapping(raw or {}, "agents")
    return AgentsConfig(
        sitl=parse_env_agents(d.get("sitl", {}), "agents.sitl"),
        hitl=parse_env_agents(d.get("hitl", {}), "agents.hitl"),
    )
