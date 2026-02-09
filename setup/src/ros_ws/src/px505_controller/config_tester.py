from __future__ import annotations

from px505_controller.config import load_config

from dataclasses import asdict, is_dataclass
from typing import Any

import json


def _to_jsonable(x: Any) -> Any:
    # dataclasses -> dict
    if is_dataclass(x):
        x = asdict(x)

    # tuples/sets -> lists (JSON doesn't have tuples)
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, set):
        return [_to_jsonable(v) for v in x]

    # dict -> dict with converted values
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # list -> list
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]

    # primitives
    return x


def pretty_json(obj: Any, indent: int = 2) -> str:
    return json.dumps(_to_jsonable(obj), indent=indent, ensure_ascii=False)

def main():
    cfg = load_config("/root/ros_ws/src/px505_controller/px505_controller/config/config.yaml")
    print(pretty_json(cfg, indent=4))
    
    print(cfg.controllers.sitl.controllers)
    print(cfg.agents.sitl.quadrotors.crazyflies)

if __name__ == "__main__":
    main()