import numpy as np
from control_framework.crazyflie_descriptor import CrazyflieDescriptor, CrazyflieState

class CustomController:

    def compute_control_action(self, crazyflie_descriptor: CrazyflieDescriptor) -> np.ndarray:
        pass