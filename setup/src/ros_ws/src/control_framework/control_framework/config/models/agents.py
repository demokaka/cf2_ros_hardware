from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict, model_validator, RootModel
from typing import Dict, Union, List, Optional, Literal, Any

class CrazyflieConfig(BaseModel):
    id: int
    uri: str
    mass: float

class EnvAgentsConfig(RootModel):
    root: Dict[str, CrazyflieConfig]

    @property
    def items(self) -> Dict[str, CrazyflieConfig]:
        return self.root

    @property
    def agents(self) -> Dict[str, CrazyflieConfig]:
        return self.root

    @property
    def id_to_name(self) -> Dict[int, str]:
        return {a.id: name for name, a in self.root.items()}

class AgentsConfig(BaseModel):
    sitl: EnvAgentsConfig
    hitl: EnvAgentsConfig