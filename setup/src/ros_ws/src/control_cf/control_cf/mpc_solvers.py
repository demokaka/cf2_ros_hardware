import numpy as np
import casadi as cas
import ray

from .control_packagecf import *
from .Bspline_conversionMatrix import *


# ----------------------------------------------------------------------------------------------------------------------
# SETUP FUNCTIONS: setup the solver only once, then call the solve method from a separate function call
# ----------------------------------------------------------------------------------------------------------------------

def setup_takeoff_solver(solver_config):
    """
    Initialize the CasADi optimization solver for computing takeoff trajectories.
    Using takeoff specific parameters (Npred_to, Q_to, R_to).
    This solver uses B-spline parameterization to generate smooth and dynamically feasible
    position, velocity, and acceleration profiles during the takeoff phase.
    The solver's solve method will return the optimal control point matrix.
    """
    n, du = 6, 3
    Ts = solver_config['Ts']
    Npred = solver_config['Npred_to']
    Q = solver_config['Qto']
    R = solver_config['Rto']
    Vc = solver_config['Vc']

    spline_deg = solver_config['spline_deg']
    n_ctrl_pts = solver_config['n_ctrl_pts']

    # THETA = solver_config['Theta']
    # THETA_1 = np.linalg.inv(THETA)
    # THETA_2 = THETA_1 @ THETA_1

    # ---- B-spline Setup ----
    knot_vec = knot_vector(spline_deg, n_ctrl_pts, [0, Ts * Npred])
    basis_funcs = b_spline_basis_functions(n_ctrl_pts, spline_deg, knot_vec)
    conv_M = bsplineConversionMatrices(n_ctrl_pts, spline_deg, knot_vec)

    # ---- Discretized System Dynamics ----
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 6))]])
    B = np.block([[np.zeros((3, 3))], [np.eye(3)]])
    A_d = np.eye(6) + Ts * A
    # B_d = B * Ts
    B_d = np.block([[Ts * Ts / 2 * np.eye(3)], [Ts * np.eye(3)]])

    solver = cas.Opti()

    P_i = solver.variable(3, n_ctrl_pts)

    xinit = solver.parameter(n, 1)
    vinit = solver.parameter(du, 1)
    xref_param = solver.parameter(n, Npred)
    vref_param = solver.parameter(du, Npred)
    psi_ref_param = solver.parameter(1, 1)

    # Precompute derivative of spline for velocity and acceleration
    P_1 = cas.mtimes(P_i, conv_M[0])
    P_2 = cas.mtimes(P_i, conv_M[1])

    # Constrain spline to start at current state (position and velocity continuity)
    b0 = np.array([f(0.0) for f in basis_funcs[0]]).reshape(-1, 1)
    b1 = np.array([f(0.0) for f in basis_funcs[1]]).reshape(-1, 1)
    pos = cas.mtimes(P_i, b0)
    vel = cas.mtimes(P_1, b1)
    solver.subject_to(cas.vertcat(pos, vel) == xinit)
    
    # Build objective: track position, velocity, and acceleration references
    objective = 0
    for k in range(Npred):
        tk = Ts * k
        b0 = np.array([f(tk) for f in basis_funcs[0]]).reshape(-1, 1)
        b1 = np.array([f(tk) for f in basis_funcs[1]]).reshape(-1, 1)
        b2 = np.array([f(tk) for f in basis_funcs[2]]).reshape(-1, 1)

        pos = cas.mtimes(P_i, b0)
        vel = cas.mtimes(P_1, b1)
        acc = cas.mtimes(P_2, b2)

        pos_error = pos - xref_param[0:3, k]
        vel_error = vel - xref_param[3:6, k]
        acc_error = acc - vref_param[0:3, k]

        objective += cas.mtimes([pos_error.T, Q[0:3, 0:3], pos_error])
        objective += cas.mtimes([vel_error.T, Q[3:6, 3:6], vel_error])
        objective += cas.mtimes([acc_error.T, R, acc_error])

        solver.subject_to(cas.mtimes(Vc['A_vc'], acc) <= Vc['b_vc'])

    solver.minimize(objective)
    opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}
    solver.solver('ipopt', opts)

    solver.set_value(vinit, np.zeros(3))

    return {
        "solver": solver, "P_i": P_i,  "xinit": xinit, "vinit": vinit,
        "xref_param": xref_param, "psi_ref_param": psi_ref_param, "vref_param": vref_param,
        "Npred": Npred, "Vc": Vc, "A_d": A_d, "B_d": B_d, "Q": Q, "R": R,
        "knot_vec": knot_vec, 'basis_funcs': basis_funcs, "conv_M": conv_M
    }


def setup_spline_solver(solver_config):
    """
    Setup the CasADi optimization solver for dynamic trajectories. 
    Almost the same as the setup_takeoff_solver, only that it is using a different set of tuning parameters (Npred, Q, R).
    It uses B-spline parameterization to generate smooth, dynamically feasible
    position, velocity, and acceleration profiles for the trajectory tracking phase.
    The solver's solve method will return the optimal control point matrix.
    """
    n, du = 6, 3
    Ts = solver_config['Ts']
    Npred = solver_config['Npred']
    Q = solver_config['Q']
    R = solver_config['R']
    Vc = solver_config['Vc']

    spline_deg = solver_config['spline_deg']
    n_ctrl_pts = solver_config['n_ctrl_pts']

    # THETA = solver_config['Theta']
    # THETA_1 = np.linalg.inv(THETA)
    # THETA_2 = THETA_1 @ THETA_1

    # ---- B-spline Setup ----
    knot_vec = knot_vector(spline_deg, n_ctrl_pts, [0, Ts * Npred])
    basis_funcs = b_spline_basis_functions(n_ctrl_pts, spline_deg, knot_vec)
    conv_M = bsplineConversionMatrices(n_ctrl_pts, spline_deg, knot_vec)

    # ---- Discretized System Dynamics ----
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 6))]])
    B = np.block([[np.zeros((3, 3))], [np.eye(3)]])
    A_d = np.eye(6) + Ts * A
    # B_d = B * Ts
    B_d = np.block([[Ts * Ts / 2 * np.eye(3)], [Ts * np.eye(3)]])

    solver = cas.Opti()

    P_i = solver.variable(3, n_ctrl_pts)

    xinit = solver.parameter(n, 1)
    vinit = solver.parameter(du, 1)
    xref_param = solver.parameter(n, Npred)
    vref_param = solver.parameter(du, Npred)
    psi_ref_param = solver.parameter(1, 1)

    # Precompute derivative of spline for velocity and acceleration
    P_1 = cas.mtimes(P_i, conv_M[0])
    P_2 = cas.mtimes(P_i, conv_M[1])

    # Constrain spline to start at current state (position and velocity continuity)
    b0 = np.array([f(0.0) for f in basis_funcs[0]]).reshape(-1, 1)
    b1 = np.array([f(0.0) for f in basis_funcs[1]]).reshape(-1, 1)
    pos = cas.mtimes(P_i, b0)
    vel = cas.mtimes(P_1, b1)
    solver.subject_to(cas.vertcat(pos, vel) == xinit)

    # Build objective: track position, velocity, and acceleration references
    objective = 0
    for k in range(Npred):
        tk = Ts * k
        b0 = np.array([f(tk) for f in basis_funcs[0]]).reshape(-1, 1)
        b1 = np.array([f(tk) for f in basis_funcs[1]]).reshape(-1, 1)
        b2 = np.array([f(tk) for f in basis_funcs[2]]).reshape(-1, 1)

        pos = cas.mtimes(P_i, b0)
        vel = cas.mtimes(P_1, b1)
        acc = cas.mtimes(P_2, b2)

        pos_error = pos - xref_param[0:3, k]
        vel_error = vel - xref_param[3:6, k]
        acc_error = acc - vref_param[0:3, k]

        objective += cas.mtimes([pos_error.T, Q[0:3, 0:3], pos_error])
        objective += cas.mtimes([vel_error.T, Q[3:6, 3:6], vel_error])
        objective += cas.mtimes([acc_error.T, R, acc_error])

        solver.subject_to(cas.mtimes(Vc['A_vc'], acc) <= Vc['b_vc'])

    solver.minimize(objective)
    opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}
    solver.solver('ipopt', opts)

    solver.set_value(vinit, np.zeros(3))

    return {
        "solver": solver, "P_i": P_i,  "xinit": xinit, "vinit": vinit,
        "xref_param": xref_param, "psi_ref_param": psi_ref_param, "vref_param": vref_param,
        "Npred": Npred, "Vc": Vc, "A_d": A_d, "B_d": B_d, "Q": Q, "R": R,
        "knot_vec": knot_vec, 'basis_funcs': basis_funcs, "conv_M": conv_M
    }


def compute_spline_control(solver_data, state_xi, pos_ref, v_ref, i):
    """
    Compute the optimal control input using the pre-configured B-spline solver.
    This function feeds current state and reference trajectories into the solver,
    solves the optimization problem, and extracts the virtual input (acceleration).

    Args:
        solver_data (dict): Output of setup_spline_solver() containing solver, parameters, etc.
        state_xi (np.ndarray): Current drone state [position, velocity, orientation, etc.]
        pos_ref (np.ndarray): Reference trajectory for position and velocity [N x 6]
        v_ref (np.ndarray): Reference trajectory for acceleration [N x 3]
        i (int): Current timestep index in the reference trajectories

    Returns:
        list: Optimal virtual control input [ax, ay, az] as floats
    """
    # ---- Extract reference trajectories for current horizon ----
    if i + solver_data["Npred"] <= len(pos_ref):
        # desired_pos = pos_ref[i:i+solver_data["Npred"], 0:3] + np.tile(self.formation_offsets[id].reshape(1, -1), (self.Npred, 1))
        desired_pos = pos_ref[i:i+solver_data["Npred"], 0:3]
        ref = np.vstack([desired_pos.T, pos_ref[i:i+solver_data["Npred"], 3:6].T])
        virt_ref = v_ref[i:i+solver_data["Npred"], 0:3].T
    else:
        # Near end of trajectory: pad with last reference point
        remaining = len(pos_ref) - i
        desired_pos = np.vstack([pos_ref[i:, 0:3], np.tile(pos_ref[-1, 0:3], (solver_data["Npred"] - remaining, 1))])  # stack the last position
        desired_vel = np.vstack([pos_ref[i:, 3:6], np.tile(pos_ref[-1, 3:6], (solver_data["Npred"] - remaining, 1))])
        virt_ref = np.vstack([v_ref[i:, 0:3], np.tile(v_ref[-1, 0:3], (solver_data["Npred"] - remaining, 1))])
        ref = np.vstack([desired_pos.T, desired_vel.T])
        virt_ref = virt_ref.T

    # ---- Set parameters in CasADi solver ----
    solver_data["solver"].set_value(solver_data["xinit"], state_xi[:, 0:6].T)
    solver_data["solver"].set_value(solver_data["xref_param"], ref)
    solver_data["solver"].set_value(solver_data["vref_param"], virt_ref)
    solver_data["solver"].set_value(solver_data["psi_ref_param"], 0)

    sol = solver_data["solver"].solve()
    P_sol = sol.value(solver_data["P_i"]) # Optimal Control Points

    # ---- Extract acceleration at t=0 from B-spline ----
    b2 = np.array([f(0.0) for f in solver_data["basis_funcs"][2]]).reshape(-1, 1)
    vopt = P_sol @ solver_data["conv_M"][1] @ b2

    return [float(vopt[0, 0]), float(vopt[1, 0]), float(vopt[2, 0])]


def setup_solver(solver_config):
    """
    Setup the CasADi optimization solver for dynamic trajectories optimizing for acceleration. 
    The solver's solve method will return the optimal acceleration directly.
    """
    # ---- Load configuration ----
    n, du = 6, 3
    Ts = solver_config['Ts']
    Npred = solver_config['Npred']
    Q = solver_config['Q']
    R = solver_config['R']
    Vc = solver_config['Vc']
    
    # ---- Discretized System Dynamics ----
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 6))]])
    B = np.block([[np.zeros((3, 3))], [np.eye(3)]])
    A_d = np.eye(6) + Ts * A
    # B_d = B * Ts
    B_d = np.block([[Ts * Ts / 2 * np.eye(3)], [Ts * np.eye(3)]])

    solver = cas.Opti()

    x = solver.variable(n, Npred + 1)
    v = solver.variable(du, Npred)
    xinit = solver.parameter(n, 1)
    vinit = solver.parameter(du, 1)
    xref_param = solver.parameter(n, Npred)
    vref_param = solver.parameter(du, Npred)
    psi_ref_param = solver.parameter(1, 1)

    # Set constraints
    solver.subject_to(x[:, 0] == xinit)
    for k in range(Npred):
        solver.subject_to(x[:, k+1] == A_d @ x[:, k] + B_d @ v[:, k])
        solver.subject_to(cas.mtimes(Vc['A_vc'], v[:, k]) <= Vc['b_vc'])

    # Set objective
    objective = 0
    for k in range(Npred):
        state_error = x[:, k] - xref_param[:, k]
        # control_effort = v[:, k] - (v[:, k-1] if k > 0 else vinit)
        control_effort = v[:, k] - vref_param[:, k]
        objective += cas.mtimes([state_error.T, Q, state_error]) + cas.mtimes([control_effort.T, R, control_effort])

    solver.minimize(objective)
    opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}
    solver.solver('ipopt', opts)

    solver.set_value(vinit, np.zeros(3))

    return {
        "solver": solver,
        "x": x, "v": v, "xinit": xinit, "vinit": vinit,
        "xref_param": xref_param, "psi_ref_param": psi_ref_param, "vref_param": vref_param,
        "Npred": Npred, "Vc": Vc, "A_d": A_d, "B_d": B_d, "Q": Q, "R": R
    }


def compute_control_real(solver_data, state_xi, pos_ref, v_ref, i, id=0):
    """
    Compute optimal control input using the standard MPC solver (no B-splines).

    Args:
        solver_data (dict): Output of setup_solver()
        state_xi (np.ndarray): Current drone state
        pos_ref (np.ndarray): Position reference trajectory for the current drone [N x 6]
        v_ref (np.ndarray): Virtual control reference for the current drone [N x 3]
        i (int): Current timestep index
        id (int, optional): Drone ID (for formation offsets, if used)

    Returns:
        list: Optimal virtual control input [ax, ay, az]
    """
    # ---- Extract reference trajectories for current horizon ----
    if i + solver_data["Npred"] <= len(pos_ref):
        # desired_pos = pos_ref[i:i+solver_data["Npred"], 0:3] + np.tile(self.formation_offsets[id].reshape(1, -1), (self.Npred, 1))
        desired_pos = pos_ref[i:i+solver_data["Npred"], 0:3]
        ref = np.vstack([desired_pos.T, pos_ref[i:i+solver_data["Npred"], 3:6].T])
        virt_ref = v_ref[i:i+solver_data["Npred"], 0:3].T
    else:
        # Near end of trajectory: pad with last reference point
        remaining = len(pos_ref) - i
        desired_pos = np.vstack([pos_ref[i:, 0:3], np.tile(pos_ref[-1, 0:3], (solver_data["Npred"] - remaining, 1))])  # stack the last position
        desired_vel = np.vstack([pos_ref[i:, 3:6], np.tile(pos_ref[-1, 3:6], (solver_data["Npred"] - remaining, 1))])
        virt_ref = np.vstack([v_ref[i:, 0:3], np.tile(v_ref[-1, 0:3], (solver_data["Npred"] - remaining, 1))])
        ref = np.vstack([desired_pos.T, desired_vel.T])
        virt_ref = virt_ref.T

    # ---- Set parameters in CasADi solver ----
    solver_data["solver"].set_value(solver_data["xinit"], state_xi[:, 0:6].T)
    solver_data["solver"].set_value(solver_data["xref_param"], ref)
    solver_data["solver"].set_value(solver_data["vref_param"], virt_ref)
    solver_data["solver"].set_value(solver_data["psi_ref_param"], 0)

    sol = solver_data["solver"].solve() 
    vopt = sol.value(solver_data["v"]) # Get the optimal acceleration
    return [float(vopt[0, 0]), float(vopt[1, 0]), float(vopt[2, 0])]


@ray.remote
class MPCWorker:
    """
    Ray Actor for running an MPC solver in parallel for a single drone.
    Each worker (actor) maintains its own solver instance, avoiding repeated setup costs.

    Attributes:
        id (int): Unique agent ID for identification.
        drone (str): Drone name or key in the states dictionary.
        uri (str): Unique resource identifier for the drone (used for reference selection).
        solver_data (dict): Contains CasADi solver and relevant parameters.
        Na (int): Number of agents in the system (from solver_config).
        Npred (int): Prediction horizon length.

    Methods:
        solve(states, pos_ref, v_ref, i):
            Solves the MPC problem at time step i for this drone.
            Returns optimal virtual input and current yaw.
    """
    def __init__(self, agent_id, drone, uri, solver_config):
        """
        Initialize the MPC worker by setting up a solver instance.

        Args:
            agent_id (int): Unique identifier for the drone agent.
            drone (str): Drone name or key in the state dictionary.
            uri (str): Unique resource identifier for this drone's data in pos_ref/v_ref.
            solver_config (dict): Configuration parameters for solver setup.
        """
        self.id = agent_id
        self.drone = drone
        self.uri = uri
        self.solver_data = setup_solver(solver_config)
        self.Na = solver_config['Na']
        self.Npred = self.solver_data['Npred']

    def solve(self, states, pos_ref, v_ref, i):
        """
        Solve the MPC optimization problem for the current drone at timestep i.

        Args:
            states (dict): Dictionary of current states for *all* drones.
            pos_ref (dict): Dictionary of reference trajectories for positions/velocities for *all* drones.
            v_ref (dict): Dictionary of virtual input reference trajectories (acceleration) for *all* drones.
            i (int): Current timestep index in the trajectory.

        Returns:
            tuple: (agent_id, [ax, ay, az], yaw)
                - agent_id (int): The drone's unique identifier.
                - [ax, ay, az] (list): Optimal virtual input (accelerations).
                - yaw (float): Current yaw angle of the drone in radians.
        """
        # ---- Extract relevant reference data for this drone ----
        pos_ref = pos_ref[self.uri]
        v_ref = v_ref[self.uri]

        state_xi = states[self.drone][:, 0:6].T
        _, _, yaw_tmp = quaternion_to_euler(states[self.drone][:, 6:10][0])

        solver = self.solver_data['solver']

        # ---- Extract reference trajectories for current horizon ----
        if i + self.Npred <= len(pos_ref):
            desired_pos = pos_ref[i:i+self.Npred, 0:3]
            ref = np.vstack([desired_pos.T, pos_ref[i:i+self.Npred, 3:6].T])
            virt_ref = v_ref[i:i+self.solver_data["Npred"], 0:3].T
        else:
            # Near end of trajectory: pad with last reference point
            remaining = len(pos_ref) - i
            desired_pos = np.vstack([pos_ref[i:, 0:3], np.tile(pos_ref[-1, 0:3], (self.Npred - remaining, 1))])  # stack the last position
            desired_vel = np.vstack([pos_ref[i:, 3:6], np.tile(pos_ref[-1, 3:6], (self.Npred - remaining, 1))])
            virt_ref = np.vstack([v_ref[i:, 0:3], np.tile(v_ref[-1, 0:3], (self.Npred - remaining, 1))])
            ref = np.vstack([desired_pos.T, desired_vel.T])
            virt_ref = virt_ref.T

        # ---- Set parameters in CasADi solver ----
        solver.set_value(self.solver_data['xinit'], state_xi[:, 0:6].T)
        solver.set_value(self.solver_data['xref_param'], ref)
        solver.set_value(self.solver_data['vref_param'], virt_ref)
        solver.set_value(self.solver_data['psi_ref_param'], 0)

        sol = solver.solve()
        vopt = sol.value(self.solver_data['v'])
        vopt = [float(vopt[0, 0]), float(vopt[1, 0]), float(vopt[2, 0])]
        return self.id, vopt, yaw_tmp


@ray.remote
class MPCWorkerCFT:
    """
    Ray Actor for running an MPC solver in parallel for a drone in a cluster formation.
    Each worker (actor) accounts for formation offsets when computing reference trajectories.

    Attributes:
        id (int): Unique agent ID for identification.
        drone (str): Drone name or key in the states dictionary.
        uri (str): Unique resource identifier for this drone's data.
        solver_data (dict): Contains CasADi solver and relevant parameters.
        Na (int): Number of agents in the system (from solver_config).
        Npred (int): Prediction horizon length.

    Methods:
        solve(states, pos_ref, v_ref, i, drone_index_in_cluster, cluster_size):
            Solves the MPC problem at time step i with formation offsets.
            Returns optimal virtual input and current yaw.
    """
    def __init__(self, agent_id, drone, uri, solver_config):
        self.id = agent_id
        self.drone = drone
        self.uri = uri
        self.solver_data = setup_solver(solver_config)
        self.Na = solver_config['Na']
        self.Npred = self.solver_data['Npred']

    def solve(self, states, pos_ref, v_ref, i, drone_index_in_cluster, cluster_size):
        """
        Solve the MPC optimization problem for this drone with formation offsets.

        Args:
            states (dict): Dictionary of current states for *all* drones.
            pos_ref (np.ndarray): Reference trajectory for position and velocity [N x 6] of the current drone.
            v_ref (np.ndarray): Reference trajectory for virtual input (acceleration) [N x 3] of the current drone.
            i (int): Current timestep index.
            drone_index_in_cluster (int): Index of this drone in its formation cluster.
            cluster_size (int): Total number of drones in the cluster.

        Returns:
            tuple: (agent_id, [ax, ay, az], yaw)
                - agent_id (int): The drone's unique identifier.
                - [ax, ay, az] (list): Optimal virtual input (accelerations).
                - yaw (float): Current yaw angle of the drone in radians.
        """
        # ---- Extract current state and orientation for the drone assigned to this actor ----
        state_xi = states[self.drone][:, 0:6].T
        _, _, yaw_tmp = quaternion_to_euler(states[self.drone][:, 6:10][0])

        # ---- Compute formation offset ----
        offsets = generate_formation_offsets(cluster_size, d=1)
        offset = offsets[drone_index_in_cluster]

        solver = self.solver_data['solver']

        if i + self.Npred <= len(pos_ref):
            desired_pos = pos_ref[i:i+self.Npred, 0:3] + offset
            ref = np.vstack([desired_pos.T, pos_ref[i:i+self.Npred, 3:6].T])
            virt_ref = v_ref[i:i+self.solver_data["Npred"], 0:3].T
        else:
            remaining = len(pos_ref) - i  # stack the last position for as many samples as is necessary
            desired_pos = np.vstack([pos_ref[i:, 0:3], np.tile(pos_ref[-1, 0:3], (self.Npred - remaining, 1))]) + offset
            desired_vel = np.vstack([pos_ref[i:, 3:6], np.tile(pos_ref[-1, 3:6], (self.Npred - remaining, 1))])
            virt_ref = np.vstack([v_ref[i:, 0:3], np.tile(v_ref[-1, 0:3], (self.Npred - remaining, 1))])
            ref = np.vstack([desired_pos.T, desired_vel.T])
            virt_ref = virt_ref.T

        solver.set_value(self.solver_data['xinit'], state_xi[:, 0:6].T)
        solver.set_value(self.solver_data['xref_param'], ref)
        solver.set_value(self.solver_data['vref_param'], virt_ref)
        solver.set_value(self.solver_data['psi_ref_param'], 0)

        sol = solver.solve()
        vopt = sol.value(self.solver_data['v'])
        vopt = [float(vopt[0, 0]), float(vopt[1, 0]), float(vopt[2, 0])]
        return self.id, vopt, yaw_tmp
