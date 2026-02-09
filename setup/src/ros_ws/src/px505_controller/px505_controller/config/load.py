from __future__ import annotations

from pathlib import Path
import yaml

from px505_controller.config.parsing.global_config import parse_global_config
from px505_controller.config.models.global_config import GlobalConfig


def load_config(path: str | Path) -> GlobalConfig:
    """
    YAML I/O only: read YAML -> parse -> validate -> GlobalConfig
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return parse_global_config(raw)
