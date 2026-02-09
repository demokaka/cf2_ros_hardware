import numpy as np
import scipy.io

"""
This script provides some functions for controlling purposes:
- Different versions of Thrust_to_PWM function:
    + Thrust_to_PWM(thrust, alpha)
    + Thrust_to_PWM_v1(thrust)
    + Thrust_to_PWM_modified(thrust, mass)
- Compute the control input for Crazyflie: 

- Feedback controller function: compute_control(v_ref, x0, xref, Kf)

To control the Crazyflie drone, we need to send a packet of control input over the simulation time.

The approach that we used is based on flatness control.This will allow us to linearize the non-linear system in closed loop
by applying a method called feedback linearization.
The result of this method is a double integrator dynamics.
"""


# FILE = open("src/Data_Drone/cft.log", "w")
# FILE_CL = open("src/Data_Drone/cluster.log", "w")


def get_cluster_mappings(latest_alloc, drone_bodies):
    cluster_to_drones = {}
    drone_to_cluster_idx = {}

    # Build a cluster -> drones map {'target' : [list of drone names]}
    for i, cluster_id in enumerate(latest_alloc):
        drone = drone_bodies[i]
        if cluster_id not in cluster_to_drones:
            cluster_to_drones[cluster_id] = []
        cluster_to_drones[cluster_id].append(drone)

    # Build a drone -> (cluster_id, index_in_cluster) map {'drone name' : (assigned target, index in the targets cluster)}
    for cluster_id in cluster_to_drones:
        for idx, drone in enumerate(cluster_to_drones[cluster_id]):
            drone_to_cluster_idx[drone] = (cluster_id, idx)

    return cluster_to_drones, drone_to_cluster_idx


def generate_formation_offsets(num_agents, d=0.5):
    offsets = np.zeros((num_agents, 3))

    if num_agents == 1:
        offsets[0] = [0, 0, 0]

    elif num_agents == 2:
        # Line formation
        offsets[0] = [0, -d/2, d - d/2]
        offsets[1] = [-d/2, 0, d - d/2]

    elif num_agents == 3:
        # Triangle formation
        offsets[0] = [0, -d/2, d - d/2]
        offsets[1] = [-d/2, d/2, d - d/2]
        offsets[2] = [d/2, d/2, d - d/2]

    elif num_agents == 4:
        # Square formation
        offsets[0] = [-d/2, -d/2, d - d/2]
        offsets[1] = [d/2, -d/2, d - d/2]
        offsets[2] = [-d/2, d/2, d - d/2]
        offsets[3] = [d/2, d/2, d - d/2]

    else:
        # Circular formation for 5+ agents
        theta = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
        for i in range(num_agents):
            offsets[i] = [d/2 * np.cos(theta[i]), d/2 * np.sin(theta[i]), d/2]

    return offsets


def fl_laws_unnormalized(v, yaw):
    """
    Feedback linearization laws to compute thrust and desired angles.
    :param v: The virtual input (accelerations).
    :param yaw: Measured yaw angle (scalar)
    :return: Unnormalized Thrust (T), Desired roll (phi_d) and pitch (theta_d) angles
    """
    g = 9.81
    m = 28e-3

    T = np.round(m * np.sqrt(v[0]**2 + v[1]**2 + (v[2] + g)**2), 5)
    phi_d = np.round(np.arcsin(m * (v[0] * np.sin(yaw) - v[1] * np.cos(yaw)) / T), 5)
    theta_d = np.round(np.arctan2(v[0] * np.cos(yaw) + v[1] * np.sin(yaw), v[2] + g), 5)

    return T, phi_d, theta_d


def Thrust_to_PWM(Thrust, alpha):
    """ Reference: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/pwm-to-thrust/ """
    pwm_signal = 65535 * (-140.5e-3*alpha + np.sqrt(140.5e-3 ** 2 - 4 * 0.409e-3 * (-0.099 - Thrust))) / (2 * 0.409e-3 * 256)
    return pwm_signal


def Thrust_to_PWM_v1(Thrust):
    """ Reference: https://www.research-collection.ethz.ch/handle/20.500.11850/214143 """
    a1 = 2.130295e-11
    a2 = 1.032633e-6
    a3 = 5.484560e-4
    m = 33.0  # quadcopter mass : from 28 to 33 gram
    g = 9.81  # gravitational acceleration
    kc = m*g/4000
    m_ratio = 1.9
    PWM_theo = (-a2 + np.sqrt(a2**2 + 4*a1*(Thrust * kc - a3)))/(2*a1*m_ratio)
    # Mapping 0-65535 to 10000-60000
    alpha = 0.7630
    beta = 10000
    pwm_signal = alpha * PWM_theo + beta
    return pwm_signal


def Thrust_to_PWM_modified(Thrust, m=33.0):
    """ Reference: https://www.research-collection.ethz.ch/handle/20.500.11850/214143 """
    a1 = 2.130295e-11
    a2 = 1.032633e-6
    a3 = 5.484560e-4
    # a2 = 1.00133e-6
    # a3 = -9.51544e-4

    g = 9.81
    kc = m * g / 4000
    PWM_signal = (-a2 + np.sqrt(a2**2 + 4*a1*(Thrust * kc - a3))) / (2*a1)
    return PWM_signal

def Thrust_to_PWM_quad(Thrust,m=33.0,PWM0=42050):
    a1 = 2.130295e-11
    a2 = 1.032633e-6
    a3 = 5.484560e-4
    K0 = a1*PWM0**2 + a2*PWM0 + a3
    pwm_signal = (-a2 + np.sqrt(a2**2 + 4*a1*(Thrust * K0 - a3)))/(2*a1) 
    return pwm_signal

def sat(x,min,max):
    if x<min:
        return min
    elif x>max:
        return max
    else:
        return x

def get_cf_input_linear(v_controls, yaw, T_coeff=23.5, desired_yaw = 0.0, alpha=1.000, mass = 33.0, bias=[0,0], PWM0=40000):
    g = 9.81
    T = np.round(np.sqrt(v_controls[0] ** 2 + v_controls[1] ** 2 + (v_controls[2] + g) ** 2), 5)
    # T = sat(T,0,10.4)
    phi = np.round(np.arcsin((v_controls[0] * np.sin(yaw) - v_controls[1] * np.cos(yaw)) / T), 5)
    theta = np.round(np.arctan((v_controls[0] * np.cos(yaw) + v_controls[1] * np.sin(yaw)) / (
                v_controls[2] + g)), 5)
    
    controls = [T, phi, theta]
    # Thrust_pwm = int(T_coeff*Thrust_to_PWM(controls[0] / g,alpha))
    # Thrust_pwm = int(T_coeff*Thrust_to_PWM_modified(controls[0]/g,mass)*alpha)
    Thrust_pwm = int(T_coeff*Thrust_to_PWM_quad(controls[0]/g,mass,PWM0)[0]*alpha)
    Thrust_pwm = sat(Thrust_pwm,0,60000)
    print('Thrust_pwm = ',Thrust_pwm)
    Roll = (controls[1] * 180) / np.pi  
    Pitch = (controls[2] * 180) / np.pi
    Yawrate = 0.05*(yaw - desired_yaw) * 180 / np.pi
    controls_cf = [Roll+bias[0], Pitch+bias[1], Yawrate, Thrust_pwm]
    return controls_cf


def compute_control(v_ref, x0, xref, Kf):
    """ Feedback controller for Crazyflie """
    v = v_ref + np.matmul(Kf, x0 - xref)
    return v


def compute_land_control(state_xi, pos_ref, Kp=np.diag([2, 2, 2]), Kd=np.diag([6, 6, 6])):
    e = pos_ref[:, 0:3] - state_xi[:, 0:3]
    e_dot = pos_ref[:, 3:6] - state_xi[:, 3:6]

    v = Kp @ e.T + Kd @ e_dot.T
    v = v.flatten()

    return [float(v[0]), float(v[1]), float(v[2])]


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
    return controls


def get_cf_input(v_controls, yaw, T_coeff=23.5, desired_yaw=0, alpha=1.000, mass=33.0, bias=[0, 0]):
    g = 9.81
    T = np.round(np.sqrt(v_controls[0] ** 2 + v_controls[1] ** 2 + (v_controls[2] + g) ** 2), 5)
    phi = np.round(np.arcsin((v_controls[0] * np.sin(yaw) - v_controls[1] * np.cos(yaw)) / T), 5)
    theta = np.round(np.arctan((v_controls[0] * np.cos(yaw) + v_controls[1] * np.sin(yaw)) / (v_controls[2] + g)), 5)
    controls = [T, phi, theta]

    # Thrust_pwm = int(T_coeff * Thrust_to_PWM(controls[0] / g, alpha))
    Thrust_pwm = int(T_coeff * Thrust_to_PWM_modified(controls[0]/g, mass) * alpha)

    Roll = (controls[1] * 180) / np.pi  # convert from radians to degrees
    Pitch = (controls[2] * 180) / np.pi
    Yawrate = 0.000 * (yaw - desired_yaw) * 180 / np.pi

    # Thrust_pwm = min(Thrust_pwm, 65000)

    controls_cf = [Roll + bias[0], Pitch + bias[1], Yawrate, Thrust_pwm]
    return controls_cf


def quaternion_to_euler(q):
    """ q = [x, y, z, w]"""
    x, y, z, w = q[0], q[1], q[2], q[3]
    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Clamping to 90 degrees
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]


def save_data(file_dest, data, ref, vref):
    data = {'data': data, 'ref': ref, 'vref': vref}
    np.save(file_dest, data)


def generate_formation_offsets(num_agents, d=0.5):
    offsets = np.zeros((num_agents, 3))

    if num_agents == 1:
        offsets[0] = [0, 0, 0]

    elif num_agents == 2:
        # Line formation
        offsets[0] = [0, d/2, d/4]
        offsets[1] = [0, -d/2, d/2]
        # offsets[1] = [-d/2, 0, d - d/2]

    elif num_agents == 3:
        # Triangle formation
        offsets[0] = [0, -d/2, d - d/2]
        offsets[1] = [-d/2, d/2, d - d/2]
        offsets[2] = [d/2, d/2, d - d/2]

    elif num_agents == 4:
        # Square formation
        offsets[0] = [-d/2, -d/2, d - d/2]
        offsets[1] = [d/2, -d/2, d - d/2]
        offsets[2] = [-d/2, d/2, d - d/2]
        offsets[3] = [d/2, d/2, d - d/2]

    else:
        # Circular formation for 5+ agents
        theta = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
        for i in range(num_agents):
            offsets[i] = [d/2 * np.cos(theta[i]), d/2 * np.sin(theta[i]), d/2]

    return offsets


class YawUnwrapper:
    def __init__(self):
        self.previous_yaw = None

    def unwrap(self, current_yaw):
        if self.previous_yaw is None:
            self.previous_yaw = current_yaw
            return current_yaw

        delta = current_yaw - self.previous_yaw

        # Adjust yaw to prevent large backward turns
        if delta > np.pi:
            current_yaw -= 2 * np.pi
        elif delta < -np.pi:
            current_yaw += 2 * np.pi

        # Ensure yaw remains forward-aligned
        if abs(current_yaw - self.previous_yaw) > np.pi / 2:
            current_yaw += np.pi  # Flip 180 degrees to maintain forward motion
            current_yaw = (current_yaw + np.pi) % (2 * np.pi) - np.pi  # Keep within [-pi, pi]

        self.previous_yaw = current_yaw
        return current_yaw
