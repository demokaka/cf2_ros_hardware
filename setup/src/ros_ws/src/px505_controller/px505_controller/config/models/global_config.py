from __future__ import annotations

from dataclasses import dataclass

from px505_controller.config.models.agents import AgentsConfig
from px505_controller.config.models.controllers import ControllersConfig


@dataclass(frozen=True)
class GlobalConfig:
    agents: AgentsConfig
    controllers: ControllersConfig
