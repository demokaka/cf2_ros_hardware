from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Crazyflie:
    name: str        # YAML key (any string)
    id: int          # numeric id
    uri: str
    mass: float


@dataclass(frozen=True)
class QuadrotorsConfig:
    crazyflies: dict[str, Crazyflie]   # by name
    id_to_name: dict[int, str]         # index for lookup/validation


@dataclass(frozen=True)
class EnvAgentsConfig:
    quadrotors: QuadrotorsConfig


@dataclass(frozen=True)
class AgentsConfig:
    sitl: EnvAgentsConfig
    hitl: EnvAgentsConfig
