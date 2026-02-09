import casadi
import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
import copy
from . import Bspline_conversionMatrix as BsplineM


from .traj_params import *
from .get_solver_traj_waypoints import *
# from Functions.traj_gen import *
from .takeoff_landing import *
from .Bspline.bspline_casadi import *

def get_ref(psi, Tsim, dt):
    k = 8  # d = k + 1, k: polynomial degree of the spline
    n_ctrl_pts = 28  # number of control points
    knot = [0, Tsim]
    g = 9.81
    # psi = 0 * np.pi / 180
    knot = BsplineM.knot_vector(k, n_ctrl_pts, knot)
    tt = np.arange(min(knot), max(knot), dt)
    bs_list = BsplineM.b_spline_basis_functions(n_ctrl_pts, k, knot)

    # Conversion matrix M
    M = BsplineM.bsplineConversionMatrices(n_ctrl_pts, k, knot)

    # Waypoints
    # W = np.array([[0, 0.2, 0.5, 0.4, 0, -0.6, -0.6, -0.6, -0.6],
    #               [0, 0, 0, 0.6, 0.7, 0.7, 0.4, -0.3, -0.6],
    #               [0.35, 0.7, 0.9, 1.1, 1.1, 1.0, 0.9, 0.7, 0.25]  # 3D test
    #               ])
    # W = np.array([[0, 0.4, 0.5, 0.4, 0, -0.6, -0.6, -0.6, -0.6],
    #               [0, 0, 0, 0.6, 0.7, 0.7, 0.4, -0.3, -0.6],
    #               [0.1, 0.25, 0.7, 1.1, 1.1, 1.0, 0.9, 0.7, 0.25]  # 3D test
    #               ])
    W = np.array([[0, 0.3, 0.5, 0.75, 0.3, 0, -0.4, -0.3, 0],
                  [0, -0.3, 0, 0.3, 0.5, 0.65, 0.4, 0.3, 0],
                  [0.4, 0.45, 0.50, 0.65, 0.85, 1.0, 0.70, 0.65, 0.45]  # 3D test
                  ])
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1])
    ctrl_pts_timestamps = np.linspace(min(knot), max(knot), n_ctrl_pts)

    ############################### Optimization problem construction ###############################
    solver = casadi.Opti()
    # Control point as optimization variable
    P = solver.variable(W.shape[0], n_ctrl_pts)

    # Objective function
    objective = 0
    P1 = casadi.mtimes(P, M[0])
    for i in range(n_ctrl_pts + 1):
        for j in range(n_ctrl_pts + 1):
            def f_lamb(t, it=i, jt=j): return bs_list[1][it](t) * bs_list[1][jt](t)
            buff_int = quad(f_lamb, min(knot), max(knot))[0]
            objective = objective + casadi.mtimes(casadi.transpose(casadi.mtimes(buff_int, P1[:, i])), P1[:, j])

    # Implementing waypoint constraints
    for i in range(W.shape[1]):
        tmp_bs = np.zeros((len(bs_list[0]), 1))
        for j in range(len(bs_list[0])):
            tmp_bs[j] = bs_list[0][j](waypoint_time_stamps[i])
        # Mathematically, casadi.mtimes(P, tmp_bs) = P * tmp_bs
        solver.subject_to(casadi.mtimes(P, tmp_bs) == W[:, i])

    # Final velocity is zero
    for i in range(W.shape[1]):
        tmp_bs = np.zeros((len(bs_list[1]), 1))
        for j in range(len(bs_list[1])):
            tmp_bs[j] = bs_list[1][j](waypoint_time_stamps[-1] - dt)
        solver.subject_to(casadi.mtimes(P1, tmp_bs) == 0)

    # Final acceleration is zero
    P2 = casadi.mtimes(P, M[1])
    for i in range(W.shape[1]):
        tmp_bs = np.zeros((len(bs_list[2]), 1))
        for j in range(len(bs_list[2])):
            tmp_bs[j] = bs_list[2][j](waypoint_time_stamps[-1] - dt)
        solver.subject_to(casadi.mtimes(P2, tmp_bs) == 0)

    solver.minimize(objective)

    solver_options = {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': 0}
    solver.solver('ipopt', solver_options)
    # ============================================================================================
    print('Generating reference ...')
    sol = solver.solve()  # Solve for the control points
    # ============================================================================================
    # Construct the result curve
    P = sol.value(P)
    print('Optimal control-points found')
    # Compute the Bspline with the solution of P
    z = []
    for i in range(P.shape[0]):
        z.append(BSpline(knot, P[i], k))

    # First derivative of the flat output
    P1 = np.array(P * M[0])
    z_d = []
    for i in range(P1.shape[0]):
        z_d.append(BSpline(knot, P1[i], k - 1))

    # Second derivative of the flat output
    P2 = np.array(P * M[1])
    z_dd = []
    for i in range(P2.shape[0]):
        z_dd.append(BSpline(knot, P2[i], k - 2))

    x = z[0](tt)
    y = z[1](tt)
    z = z[2](tt)
    dx = z_d[0](tt)
    dy = z_d[1](tt)
    dz = z_d[2](tt)
    ddx = z_dd[0](tt)
    ddy = z_dd[1](tt)
    ddz = z_dd[2](tt)

    # ref = np.stack([x, y, z, dx, dy, dz, ddx, ddy, ddz])
    v_ref = np.stack([ddx, ddy, ddz])
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * np.sin(psi) - ddy * np.cos(psi)) / thrust)
    theta = np.arctan((ddx * np.cos(psi) + ddy * np.sin(psi)) / (ddz + g))

    # print('Reference information:')
    # print('Max Thrust: {txta}g (m/s^2), Min Thrust: {txtb}g (m/s^2)'.format(txta=round(max(thrust) / g, 2),
    #                                                                         txtb=round(min(thrust) / g, 2)))
    # print('Max Roll  : {txta}  (deg),   Min Roll  : {txtb}  (deg)'.format(txta=round(max(phi) * 180 / np.pi, 2),
    #                                                                       txtb=round(min(phi) * 180 / np.pi, 2)))
    # print('Max Pitch : {txta}  (deg),   Min Pitch : {txtb}  (deg)'.format(txta=round(max(theta) * 180 / np.pi, 2),
    #                                                                       txtb=round(min(theta) * 180 / np.pi, 2)))
    ref = {"trajectory": (np.round(np.stack([x, y, z, dx, dy, dz]), 3)).transpose(),
           "time_step": tt,
           "thrust": thrust,
           "phi": phi,
           "theta": theta,
           "Nsim": tt.shape[0],
           "v_ref": v_ref.transpose()}

    return ref


def get_ref_setpoints(psi, Tsim, dt, version=1):
    knot = [0, Tsim]
    g = 9.81
    tt = np.arange(min(knot), max(knot), dt)
    if version == 1:
        W = np.array([[0, 0.3, 0.6, 0.6, 0.3, 0, -0.3, -0.3, 0],
                      [0, -0.3, 0, 0.3, 0.6, 0.6, 0.3, 0, 0],
                      [0.35, 0.4, 0.75, 0.8, 0.8, 0.8, 0.8, 0.5, 0.35]  # 3D test
                      ])
    elif version == 2:
        W = np.array([[0, 0.6, 0.3, -0.3, 0],
                      [0, 0, 0.6, 0.3, 0],
                      [0.35, 0.75, 0.8, 0.8, 0.35]  # 3D test
                      ])
    elif version == 3:
        W = np.array([[0.6, 0.6],
                      [0.6, 0.6],
                      [0.8, 0.8]  # 3D test
                      ])
    elif version == 7:
        W = np.array([[0, 0],
                      [0, 0],
                      [0.8, 0.8]  # 3D test
                      ])
    k_pass = 1
    ref_tmp = np.empty((0, 3))
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(W[:, i_tmp])
        while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1

    ref_full = np.block([
        [ref_tmp, ref_tmp * 0]
    ])
    v_ref = 0 * ref_tmp.transpose()
    ddx, ddy, ddz = v_ref[0, :], v_ref[1, :], v_ref[2, :]
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * np.sin(psi) - ddy * np.cos(psi)) / thrust)
    theta = np.arctan((ddx * np.cos(psi) + ddy * np.sin(psi)) / (ddz + g))
    ref = {
        "trajectory": ref_full,
        "time_step": tt,
        "thrust": thrust,
        "phi": phi,
        "theta": theta,
        "Nsim": tt.shape[0],
        "v_ref": v_ref.transpose()}

    return ref


# rref = get_ref(0, 30, 0.1)

# # rref2 = copy.deepcopy(rref)
# # rref2["trajectory"] = np.copy(rref["trajectory"])
# # rref2["trajectory"][:, 0] += 1
# # rref2["trajectory"][:, 1] += 1
# # rref2["trajectory"][:, 2] += 1

# # fig2 = plt.figure()
# # for i in range(3):
# #     ax2 = fig2.add_subplot(3, 1, i + 1)
# #     it = i + 0
# #     ax2.plot(rref["time_step"], rref['trajectory'][:, it])
# #     ax2.grid(True)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection="3d")

# ax.plot(rref["trajectory"][:, 0], rref["trajectory"][:, 1], rref["trajectory"][:, 2], label="Trajectory 1")
# # ax.plot(rref2["trajectory"][:, 0], rref2["trajectory"][:, 1], rref2["trajectory"][:, 2], label="Trajectory 2")

# # # Labels and legend
# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")
# # ax.legend()

# plt.show()

def get_ref_trc():
    nbr_ctrl_pts = 12
    n = nbr_ctrl_pts - 1
    deg = 3
    Tf = 30                 # final time
    knot_endpoints = [0,Tf]

    [bs, knot, x] = bsplines_casadi(n, deg, knot_endpoints)
    [M, Sd] = bsplineConversionMatrices(n, deg, knot)
    bs_index = list(bs.keys())
    traj = BsplineTrajParams(nbr_ctrl_pts=nbr_ctrl_pts, deg = deg, bs=bs, bs_index=bs_index, M=M, knot=knot, Tf=Tf, h=0.1)

    dimension = 3

    # declare parameters
    height = 0.8

    ### Leader's reference ###
    # initial point
    pinit_leader = np.array([-1.0, -1.5, height])
    pinit_leader = pinit_leader.reshape(-1,1)
    vinit_leader = np.zeros((3,1))

    # intermediate points
    pinter_leader = np.array([[0.5, 0, height],
                                [1.5, 0.0, height],
                                [1.5, -1.0, height]]).transpose()

    vinter_leader = np.zeros((3,3))

    vinter_leader[0,0] = 0.1
    vinter_leader[1,1] = -0.1
    vinter_leader[0,2] = -0.1
    # vinter_leader[1,0] = 0.1


    # final point
    pf_leader = np.array([0.5, 0,  height])
    pf_leader = pf_leader.reshape(-1,1)

    vf_leader = np.zeros((3,1))

    W_leader = np.hstack((pinit_leader,pinter_leader,pf_leader))
    print(W_leader)
    Wvel_leader = np.hstack((vinit_leader,vinter_leader,vf_leader))
    print(Wvel_leader)

    nbr_wps = np.size(W_leader, 1)

    knot_wp = np.linspace(0, traj.Tf, nbr_wps)

    [solver, solver_vars] = get_solver_traj_waypoints(traj=traj, dimension=dimension, nbr_wps=nbr_wps)

    solver.set_value(solver_vars['W'], W_leader)
    solver.set_value(solver_vars['Wvel'], Wvel_leader)
    solver.set_value(solver_vars['knot_wp'], knot_wp)
    print(knot_wp)


    tic = time.time()
    sol = solver.solve()  # Solve for the control points
    toc = time.time()
    Elapsed_time = toc - tic
    print('Elapsed time for solving leader reference: ', Elapsed_time, '[second]')

    P = sol.value(solver_vars['P'])
    print(P)



    ctrl_pts = np.asarray(P)
    ctrl_pts_1 = ctrl_pts @ traj.M[0]
    ctrl_pts_2 = ctrl_pts @ traj.M[1]


    traj_tt = ctrl_pts @ traj.bs_eval        # trajectory
    vel_tt = ctrl_pts_1 @ traj.bs_eval1     # velocity - 1st derivative
    accel_tt = ctrl_pts_2 @ traj.bs_eval2   # acceleration - 2nd derivative

    ref_full = np.vstack((traj_tt, vel_tt))
    tt = traj.tt
    psi=0.0
    ddx, ddy, ddz = accel_tt[0, :], accel_tt[1, :], accel_tt[2, :]
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * sin(psi) - ddy * cos(psi)) / thrust)
    theta = np.arctan((ddx * cos(psi) + ddy * sin(psi)) / (ddz + g))
    ref = {}
    ref = {
        "trajectory": ref_full,
        "time_step": tt,
        "thrust": thrust,
        "phi": phi,
        "theta": theta,
        "Nsim": tt.shape[0],
        "v_ref": accel_tt}
