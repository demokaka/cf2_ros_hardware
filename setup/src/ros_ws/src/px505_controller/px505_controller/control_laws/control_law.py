import numpy as np
from px505_controller.control_laws.models import StateSpaceModel

class ControlLawSetting:
    """
    Base class for control law settings.
    Attributes:
        dt (float): Time step for the control law.
        model (StateSpaceModel): The state-space model used in the control law.
        
    This class should be extended by specific control law settings.
    """
    model: StateSpaceModel

    def __init__(self, model: StateSpaceModel):
        self.model = model

class ControlLaw:
    """
    Base class for control laws.
    Attributes:
        settings (ControlLawSetting): Settings for the control law.
    """
    def __init__(self, settings: ControlLawSetting):
        self.settings: ControlLawSetting = settings
        
    def compute_control_action(self, current_state: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
        """
        Base method to compute control action.
        Should be overridden by subclasses.
        """
        
        pass
    
    def reset(self):
        """
        Base method to compute control action.
        Should be overridden by subclasses.
        """
        
        pass