import numpy as np
from .control_packagecf import *


class PIDController:
    def __init__(self, Kp, Kd, Ki, Ts):
        self.Ts = Ts
        self.A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 6))]])
        self.B = np.block([[np.zeros((3, 3))], [np.eye(3)]])
        self.A_d = np.eye(6) + self.Ts * self.A
        self.B_d = self.B * self.Ts

        self.Kp, self.Kd, self.Ki = Kp, Kd, Ki
        self.integral_limit = 20
        self.integral_error = np.zeros(3)

    def compute_control_real(self, state_xi, pos_ref, i, id=0):
        e = pos_ref[:, 0:3] - state_xi[:, 0:3]
        e_dot = pos_ref[:, 3:6] - state_xi[:, 3:6]
        # self.e_int[id] = np.clip(self.e_int[id] + e * self.Ts, -self.integral_limit, self.integral_limit)

        v = self.Kp @ e.T + self.Kd @ e_dot.T
        v = v.flatten()

        # if self.Ki is None:
        #     v = self.Kp @ e + self.Kd @ e_dot
        # else:
        #     v = self.Kp @ e + self.Kd @ e_dot + self.Ki @ self.e_int[id]

        # self.vsim[id][:, i] = v.T

        return [float(v[0]), float(v[1]), float(v[2])]
