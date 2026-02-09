import numpy as np

def get_real_input(v_controls, yaw):
    """
    Feedback linearization laws to compute the real input from the virtual one (thrust and desired angles).
    :param v_controls: The virtual input (accelerations).
    :param yaw: Measured yaw angle (scalar, radians)
    :return: Normalized Thrust (T), Desired roll (phi_d) and pitch (theta_d) angles
    """
    g = 9.81
    T = np.round(np.sqrt(v_controls[0] ** 2 + v_controls[1] ** 2 + (v_controls[2] + g) ** 2), 5)
    phi = np.round(np.arcsin((v_controls[0] * np.sin(yaw) - v_controls[1] * np.cos(yaw)) / T), 5)
    theta = np.round(np.arctan((v_controls[0] * np.cos(yaw) + v_controls[1] * np.sin(yaw)) / (v_controls[2] + g)), 5)
    controls = [T, phi, theta]

    controls_cf = [phi, theta, 0, T]
    return controls_cf




class TrackingLQR:
    def __init__(self):
        self.solver_locked = False
        self.K_lqr =-2.0 * np.array([[2.5, 0, 0, 1.5, 0, 0],
                            [0, 2.5, 0, 0, 1.5, 0],
                            [0, 0, 2.5, 0, 0, 1.5]])

    def solve_lqr(self, x0, xref, vref):
        if self.solver_locked:
            return 
        self.solver_locked = True
        # LQR computation here
        v = vref + np.matmul(self.K_lqr, x0 - xref)  # compute control input
        control_input = get_real_input(v, 0)  # feedback linearization
        self.solver_locked = False
        
        return control_input
