from ..traj_params import *
from ..get_solver_traj_waypoints import *
# from Functions.traj_gen import *
from ..Bspline.bspline_casadi import *
# time library to measure time executed
import time
import numpy as np

g = 9.81

def get_ref_trc(Tf, Ts=0.1):
    nbr_ctrl_pts = 12
    n = nbr_ctrl_pts - 1
    deg = 3
    # Tf = 30                 # final time
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

    return ref

def generate_pap_traj(W, Wvel, knot_wp, traj, dimension = 3):
    g = 9.81
    nbr_wps = np.size(W, 1)
    [solver, solver_vars] = get_solver_traj_waypoints(traj=traj, dimension=dimension, nbr_wps=nbr_wps)

    solver.set_value(solver_vars['W'], W)
    solver.set_value(solver_vars['Wvel'], Wvel)
    solver.set_value(solver_vars['knot_wp'], knot_wp)

    tic = time.time()
    sol = solver.solve()  # Solve for the control points
    toc = time.time()

    Elapsed_time = toc - tic
    print('Elapsed time for solving reference: ', Elapsed_time, '[second]')

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
    return P, ref

def get_ref_trc_leader(height = 0.8):
    nbr_ctrl_pts = 20
    n = nbr_ctrl_pts - 1
    deg = 3
    Tf = 30                 # final time
    knot_endpoints = [0,Tf]

    [bs, knot, x] = bsplines_casadi(n, deg, knot_endpoints)
    [M, Sd] = bsplineConversionMatrices(n, deg, knot)
    bs_index = list(bs.keys())
    traj = BsplineTrajParams(nbr_ctrl_pts=nbr_ctrl_pts, deg = deg, bs=bs, bs_index=bs_index, M=M, knot=knot, Tf=Tf, h=0.1)
    

    ### Leader's reference ###
    # initial point
    # pinit_leader = np.array([-1.0, -1.5, height]) # old
    # pinit_leader = np.array([-0.5, -1.0, height])   # new: ez
    pinit_leader = np.array([-0.6, -1.2, height])   # new: ez + big
    pinit_leader = pinit_leader.reshape(-1,1)
    print(pinit_leader)
    vinit_leader = np.zeros((3,1))

    # intermediate points
    # pinter_leader = np.array([[0.5, 0, height],                   # old
    #                             [1.5, 0.0, height],
    #                             [1.5, -1.0, height]]).transpose()
    pinter_leader = np.array([[0.6, 0, height],                     # new: ez
                                [1.2, 0.0, height],
                                [1.2, -1.2, height]]).transpose()
    print(pinter_leader)

    vinter_leader = np.zeros((3,3))
    print(vinter_leader)
    vinter_leader[0,0] = 0.1
    vinter_leader[1,1] = -0.1
    # vinter_leader[1,1] = -0.0
    vinter_leader[0,2] = -0.1
    # vinter_leader[1,0] = 0.1
    print(vinter_leader)

    # final point
    # pf_leader = np.array([0.5, 0,  height]) # old ez
    # pf_leader = np.array([-0.5, -1.0,  height]) # new: hard
    pf_leader = np.array([-0.6, -1.2,  height]) # new: hard + big
    pf_leader = pf_leader.reshape(-1,1)
    print(pf_leader)
    vf_leader = np.zeros((3,1))


    W_leader = np.hstack((pinit_leader,pinter_leader,pf_leader))
    print(W_leader)
    Wvel_leader = np.hstack((vinit_leader,vinter_leader,vf_leader))
    print(Wvel_leader)

    nbr_wps = np.size(W_leader, 1)

    knot_wp = np.linspace(0, traj.Tf, nbr_wps)

    P_leader, ref = generate_pap_traj(W_leader, Wvel_leader, knot_wp, traj)

    return P_leader, ref

def get_ref_trc_ex(height = 0.8):
    nbr_ctrl_pts = 20
    n = nbr_ctrl_pts - 1
    deg = 3
    Tf = 30                 # final time
    knot_endpoints = [0,Tf]

    [bs, knot, x] = bsplines_casadi(n, deg, knot_endpoints)
    [M, Sd] = bsplineConversionMatrices(n, deg, knot)
    bs_index = list(bs.keys())
    traj = BsplineTrajParams(nbr_ctrl_pts=nbr_ctrl_pts, deg = deg, bs=bs, bs_index=bs_index, M=M, knot=knot, Tf=Tf, h=0.1)
    ### External agent's reference ###
    # initial point
    # pinit_ex = np.array([-1.5, 1.5, height])          # old
    # pinit_ex = np.array([-1.0, 1.0, height])            # new: ez
    pinit_ex = np.array([-1.2, 1.2, height])            # new: ez + big
    pinit_ex = pinit_ex.reshape(-1,1)
    print(pinit_ex)
    vinit_ex = np.zeros((3,1))

    # intermediate points
    # pinter_ex = np.array([[0, 0.5, height],             # old: hard
    #                     [1.0, 1.0, height],
    #                     [0, 1.5, height]]).transpose()
    pinter_ex = np.array([[0, 0.6, height],             # new: ez
                        [1.0, 1.0, height],
                        [0, 1.2, height]]).transpose()
    print(pinter_ex)

    vinter_ex = np.zeros((3,3))
    print(vinter_ex)
    vinter_ex[0,0] = 0.1
    vinter_ex[1,1] = 0.1
    # vinter_ex[1,1] = 0.0
    vinter_ex[0,2] = -0.1
    # vinter_leader[1,0] = 0.1
    print(vinter_ex)

    # final point
    # pf_ex = np.array([-1.5, 1.5,  height])            # old
    # pf_ex = np.array([-1.0, 1.0,  height])        # new: ez
    pf_ex = np.array([-1.2, 1.2,  height])          # new: ez + big
    pf_ex = pf_ex.reshape(-1,1)
    print(pf_ex)
    vf_ex = np.zeros((3,1))

    W_ex = np.hstack((pinit_ex,pinter_ex,pf_ex))
    print(W_ex)
    Wvel_ex = np.hstack((vinit_ex,vinter_ex,vf_ex))
    print(Wvel_ex)

    nbr_wps = np.size(W_ex, 1)
    knot_wp = np.linspace(0, traj.Tf, nbr_wps)
    P_ex, ref_ex = generate_pap_traj(W_ex, Wvel_ex, knot_wp, traj)
    return P_ex, ref_ex

if __name__ == "__main__":
    # ref_full = get_ref_trc(Tf=30, Ts=0.1)
    ref, vref = {}, {}
    
    P_leader, ref_full = get_ref_trc_leader(height=0.8)
    P_ex, ref_full_ex = get_ref_trc_ex(height=0.8)
    g0 = 0.6
    # gap = np.zeros((3, nbr_agents))
    gap = np.zeros((3, 4))
    # self.gap[0,:] = [-self.g0, 0, -self.g0, 0]
    # self.gap[1,:] = [self.g0, self.g0, 0, 0]
    # self.gap[2,:] = [0, 0, 0, 0]
    gap[0,:] = [0, -g0, 0, -g0]
    gap[1,:] = [0, 0, g0, g0]
    gap[2,:] = [0, 0, 0, 0]

    gap_vel = np.zeros((3, 4))
    gap_acc = np.zeros((3, 4))
    ref[0] = ref_full["trajectory"]
    vref[0] = ref_full["v_ref"]
    ref_ex = ref_full_ex["trajectory"]
    vref_ex = ref_full_ex["v_ref"]
    for i in range(1, 4):
        if i <= 2:
            ref[i] = ref_full["trajectory"] + np.vstack((gap[:, [i]], np.zeros((3, 1))))
            vref[i] = ref_full["v_ref"]
        else:
            ref[i] = ref_ex
            vref[i] = vref_ex
    print(ref_full['time_step'],ref_full['trajectory'])
    print(len(ref_full['time_step']), len(ref_full['trajectory']))
    import matplotlib.pyplot as plt

    view_3Dtraj = 1     # plot 3D trajectory
    view_pos = 1
    view_vel = 1
    view_acc = 1
    # Position plot
    if view_pos:
        fig_pos = plt.figure("Position plot",)
        dimension_traj = 3
        pos_labels = ['x','y','z']
        pos_ref_plot = {}
        # pos_plot = {}

        for m in range(dimension_traj):
            ax = fig_pos.add_subplot(dimension_traj,1,m+1)
            ax.grid(True)

            pos_ref_plot[m], = ax.plot(ref_full['time_step'], ref_full['trajectory'][m,:])
            for i in range(1,4):
                ax.plot(ref_full['time_step'], ref[i][m,:])
            plt.setp(pos_ref_plot[m], color='m', linestyle='--', linewidth=3.5)

            # pos_plot[m] = {}
            # for i in range(nbr_agents):
            #     pos_plot[m][i], = ax.plot(ref['time_step'], simulator['x_sim'][m,i,:])
            #     plt.setp(pos_plot[m][i], color=colors[i], linestyle='-', linewidth=3.0)
                
            if m==0:
                ax.set_title(r"Positions on $x$, $y$ and $z$-axis", usetex=False, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=False, fontsize=12)
            ax.set_ylabel(f"${pos_labels[m]}(t)$", usetex=False, fontsize=12)

    # Velocity plot
    if view_vel:
        fig_vel = plt.figure("Velocity plot",)
        dimension_traj = 3
        vel_labels = ['v_{x}','v_{y}','v_{z}']
        vel_ref_plot = {}
        # vel_plot = {}

        for j in range(dimension_traj):
            ax = fig_vel.add_subplot(dimension_traj,1,j+1)
            ax.grid(True)

            vel_ref_plot[j], = ax.plot(ref_full['time_step'], ref_full['trajectory'][j+3,:])
            for i in range(1,4):
                ax.plot(ref_full['time_step'], ref[i][j+3,:])
            plt.setp(vel_ref_plot[j], color='m', linestyle='--', linewidth=3.5)

            # vel_plot[j] = {}
            # for i in range(nbr_agents):
            #     vel_plot[j][i], = ax.plot(ref['time_step'], simulator['x_sim'][j+3,i,:])
            #     plt.setp(vel_plot[j][i], color=colors[i], linestyle='-', linewidth=3.0)

            if j==0:
                ax.set_title(r"Velocity on $x$, $y$ and $z$-axis", usetex=False, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=False, fontsize=12)
            ax.set_ylabel(f"${vel_labels[j]}(t)$", usetex=False, fontsize=12)

    # Acceleration plot
    if view_acc:
        fig_acc = plt.figure("Acceleration plot",)
        dimension_traj = 3
        acc_labels = ['a_{x}','a_{y}','a_{z}']
        acc_ref_plot = {}
        # acc_plot = {}

        for j in range(dimension_traj):
            ax = fig_acc.add_subplot(dimension_traj,1,j+1)
            ax.grid(True)

            acc_ref_plot[j], = ax.plot(ref_full['time_step'], ref_full['v_ref'][j,:])
            for i in range(1,4):
                ax.plot(ref_full['time_step'], vref[i][j,:])
            plt.setp(acc_ref_plot[j], color='m', linestyle='--', linewidth=3.5)

            # acc_plot[j] = {}
            # for i in range(nbr_agents):
            #     acc_plot[j][i], = ax.plot(ref['time_step'][1:len(ref['time_step'])], simulator['u_sim'][j,i,:])
            #     plt.setp(acc_plot[j][i], color=colors[i], linestyle='-', linewidth=3.0)

            if j==0:
                ax.set_title(r"Acceleration on $x$, $y$ and $z$-axis", usetex=False, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=False, fontsize=12)
            ax.set_ylabel(f"${acc_labels[j]}(t)$", usetex=False, fontsize=12)

    # 3D trajectory
    if view_3Dtraj:
        # Create a 3D plot
        fig_3D = plt.figure("3D Trajectory plot",)
        ax = fig_3D.add_subplot(111, projection='3d')
        traj_3D_plot, = ax.plot(ref_full['trajectory'][0,:], ref_full['trajectory'][1,:], ref_full['trajectory'][2,:])

        for i in range(1,4):
            ax.plot(ref[i][0,:],ref[i][1,:],ref[i][2,:])
        plt.setp(traj_3D_plot, color='b', linestyle='-', linewidth=2.5)

        ## external plot #######
        # traj_3D_plot_ex, = ax.plot(traj_tt_ex[0,:], traj_tt_ex[1,:], traj_tt_ex[2,:])

        # plt.setp(traj_3D_plot_ex, color='c', linestyle='-', linewidth=2.5)
        ########################

        # Change azimuth and elevation view
        ax.view_init(elev=20, azim=-80)

        # Set labels and title
        ax.set_xlabel(r"$x(m)$", usetex=False, fontsize=12)
        ax.set_ylabel(r"$y(m)$", usetex=False, fontsize=12)
        ax.set_zlabel(r"$z(m)$", usetex=False, fontsize=12)
        ax.set_title(r"3D Trajectory", usetex=False, fontsize=14)    
    plt.show()
