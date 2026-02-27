from control_framework.config.models.global_config import GlobalConfig

import yaml
import os

from ament_index_python import get_package_share_directory

CONFIG_PATH =  os.path.join(get_package_share_directory('control_framework'), 'config.yaml')

def load_config(path: str = CONFIG_PATH) -> GlobalConfig:
    with open(path, 'r') as f:
        raw_data = yaml.safe_load(f)
    
    return GlobalConfig.model_validate(raw_data)

__all__ = ["load_config", "GlobalConfig"]
