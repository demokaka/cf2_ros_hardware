from __future__ import annotations
from pydantic import BaseModel, model_validator
from control_framework.config.models.controllers import ControllersConfig, EnvControllersConfig
from control_framework.config.models.agents import AgentsConfig, EnvAgentsConfig

class GlobalConfig(BaseModel):
    agents: AgentsConfig
    controllers: ControllersConfig

    @model_validator(mode='after')
    def validate_pairings(self) -> GlobalConfig:
        for env in ['sitl', 'hitl']:
            agent_cfg: EnvAgentsConfig = getattr(self.agents, env)
            ctrl_cfg: EnvControllersConfig = getattr(self.controllers, env)
            
            id_map = agent_cfg.id_to_name
            controlled_ids = []
            
            for c_name, ctrl in ctrl_cfg.items.items():
                if ctrl.controlled_agent_id not in id_map:
                    raise ValueError(
                        f"{env}: Controller '{c_name}' targets ID {ctrl.controlled_agent_id}, "
                        f"but it doesn't exist in agents.{env}"
                    )
                controlled_ids.append(ctrl.controlled_agent_id)
            
            if len(controlled_ids) != len(set(controlled_ids)):
                raise ValueError(f"{env}: Multiple controllers assigned to the same agent ID.")
        return self