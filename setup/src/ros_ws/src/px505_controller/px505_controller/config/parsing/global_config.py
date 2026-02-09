from __future__ import annotations

from typing import Any

from px505_controller.config.models.global_config import GlobalConfig
from px505_controller.config.parsing.helpers import require_mapping
from px505_controller.config.parsing.agents import parse_agents
from px505_controller.config.parsing.controllers import parse_controllers
from px505_controller.config.validation.pairing import validate_1to1_pairing

def parse_global_config(raw: Any) -> GlobalConfig:
    root = require_mapping(raw, "root")

    agents = parse_agents(root.get("agents", {}))
    controllers = parse_controllers(root.get("controllers", {}))

    cfg = GlobalConfig(agents=agents, controllers=controllers)

    validate_1to1_pairing("sitl", cfg.agents.sitl, cfg.controllers.sitl)
    validate_1to1_pairing("hitl", cfg.agents.hitl, cfg.controllers.hitl)

    return cfg
