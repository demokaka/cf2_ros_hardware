from __future__ import annotations

from px505_controller.config.models.agents import EnvAgentsConfig
from px505_controller.config.models.controllers import EnvControllersConfig, PIDController

def validate_1to1_pairing(env: str, agents: EnvAgentsConfig, controllers: EnvControllersConfig) -> None:
    id_to_name = agents.quadrotors.id_to_name

    controlled_ids: list[int] = []
    for ctrl_name, ctrl in controllers.controllers.items():
        # every controller has controlled_agent_id in this design
        controlled_id = ctrl.controlled_agent_id  # type: ignore[attr-defined]

        if controlled_id not in id_to_name:
            raise ValueError(
                f"{env}: controller {ctrl_name!r} controls agent id {controlled_id}, "
                f"but agents.{env}.quadrotors.crazyflies has no such id"
            )
        controlled_ids.append(controlled_id)

    # 1↔1: no duplicates
    duplicates = {i for i in controlled_ids if controlled_ids.count(i) > 1}
    if duplicates:
        dup_list = ", ".join(str(x) for x in sorted(duplicates))
        raise ValueError(f"{env}: multiple controllers assigned to the same agent id(s): {dup_list}")
