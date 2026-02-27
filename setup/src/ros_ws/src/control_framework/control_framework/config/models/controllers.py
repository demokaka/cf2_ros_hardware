from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict, RootModel
from typing import Dict, Union, List, Optional, Literal, Any

class BaseControllerParams(BaseModel):
    model_config = ConfigDict(extra='allow') 
    dt: int
    u_min: Optional[List[float]] = None
    u_max: Optional[List[float]] = None

class PIDParameters(BaseControllerParams):
    Kp: List[float]
    Ki: List[float]
    Kd: List[float]

class LQRParameters(BaseControllerParams):
    Q: List[float]
    R: List[float]

class MPCParameters(LQRParameters):
    Qf: List[float]
    horizon: int
    x_min: Optional[List[float]] = None
    x_max: Optional[List[float]] = None

# --- Controller Types ---
class PIDControllerConfig(BaseModel):
    controller_type: Literal["PID"]
    controlled_agent_id: int
    parameters: PIDParameters

class LQRControllerConfig(BaseModel):
    controller_type: Literal["LQR"]
    controlled_agent_id: int
    parameters: LQRParameters

class MPCControllerConfig(BaseModel):
    controller_type: Literal["MPC"]
    controlled_agent_id: int
    parameters: MPCParameters

class CustomControllerConfig(BaseModel):
    controller_type: Literal["Custom"]
    controlled_agent_id: int
    parameters: Dict[str, Any]

ControllerConfig = Union[PIDControllerConfig, LQRControllerConfig, MPCControllerConfig, CustomControllerConfig]

class EnvControllersConfig(RootModel):
    root: Dict[str, ControllerConfig]
    
    @property
    def items(self) -> Dict[str, ControllerConfig]:
        return self.root

class ControllersConfig(BaseModel):
    sitl: EnvControllersConfig
    hitl: EnvControllersConfig