from __future__ import annotations
import json
from control_framework.config import load_config

def main():    
    try:
        cfg = load_config()
    except Exception as e:
        print(f"Config Validation Failed:\n{e}")
        return

    # 1. Print the whole thing as a clean JSON
    print("--- Global Config JSON Dump ---")
    print(json.dumps(cfg.model_dump(), indent=4))
    
    # 2. Test SITL Controllers
    print("\n--- SITL Controllers ---")
    sitl_ctrls = cfg.controllers.sitl.items  # No .root!
    if not sitl_ctrls:
        print("No controllers defined for SITL.")
    for name, ctrl in sitl_ctrls.items():
        print(f"[{name}] Type: {ctrl.controller_type}, Targets Agent: {ctrl.controlled_agent_id}")

    # 3. Test SITL Agents
    print("\n--- SITL Agents ---")
    sitl_agents = cfg.agents.sitl.items
    for name, agent in sitl_agents.items():
        print(f"[{name}] ID: {agent.id}, Mass: {agent.mass}g")

    # 4. Test ID Map property
    print("\n--- ID Mapping Check ---")
    print(f"SITL ID -> Name: {cfg.agents.sitl.id_to_name}")

if __name__ == "__main__":
    main()