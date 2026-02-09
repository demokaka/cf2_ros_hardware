import numpy as np

class StateSpaceModel:
    """
    Base class for state-space models.
    Attributes:
        A (np.ndarray): State matrix.
        B (np.ndarray): Input matrix.
        C (np.ndarray): Output matrix.
        D (np.ndarray): Feedthrough matrix.
        dt (float): Time step.
        x (np.ndarray): State vector.
        u (np.ndarray): Input vector.            
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, dt: float):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt
        
        self.x = np.zeros((A.shape[0],))  # Initialize state vector
        self.u = np.zeros((B.shape[1],))  # Initialize input vector
        
    def propagate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Propagate the state using the state-space model.
        Args:
            x (np.ndarray): Current state vector.
            u (np.ndarray): Control input vector.
        Returns:
            np.ndarray: Next state vector.
        """        
        self.x = self.A @ x + self.B @ u
        return self.x
    
    def output(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the output using the state-space model.
        Args:
            x (np.ndarray): Current state vector.
            u (np.ndarray): Control input vector.
        Returns:
            np.ndarray: Output vector.
        """
        return self.C @ x + self.D @ u
    
class DoubleIntegrator3DOF(StateSpaceModel):    
    """
    3-DOF Double Integrator State-Space Model.
    Derived from the base StateSpaceModel class.
    """
    @staticmethod
    def from_dt(dt: float) -> "DoubleIntegrator3DOF":
        A = np.array([[1, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        
        ht = 0.5 * dt**2
        B = np.array([[ht, 0, 0],
                      [0, ht, 0],
                      [0, 0, ht],
                      [dt, 0, 0],
                      [0, dt, 0],
                      [0, 0, dt]])
        
        C = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]])
        
        D = np.zeros((6,3))
        
        return DoubleIntegrator3DOF(A=A, B=B, C=C, D=D, dt=dt)
    
class DoubleIntegrator2DOF(StateSpaceModel):
    """
    2-DOF Double Integrator State-Space Model.
    Derived from the base StateSpaceModel class.
    """
    @staticmethod
    def from_dt(dt: float) -> "DoubleIntegrator2DOF":
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        ht = 0.5 * dt**2
        B = np.array([[ht, 0],
                      [0, ht],
                      [dt, 0],
                      [0, dt]])

        C = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        
        D = np.zeros((4,2))
        
        return DoubleIntegrator2DOF(A=A, B=B, C=C, D=D, dt=dt)