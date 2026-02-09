import casadi
from scipy.integrate import quad
import numpy as np

def get_solver_traj_waypoints(traj, dimension, nbr_wps):

    """
    Inputs:
        - traj: (dict) trajectory parameters
        - dimension: (int) 2 or 3 indicate 2D or 3D space
        - nbr_wps: number of waypoints
    Outputs:
        - solver: casadi.Opti() object as optimizer
        - solver_vars: casadi variables and parameters of the solver object
    """

    nbr_ctrl_pts = traj.nbr_ctrl_pts            # take out number of control points
    n = nbr_ctrl_pts - 1                        # cardinality

    solver = casadi.Opti()
    
    # Declare variables 
    P = solver.variable(dimension, n+1)        # control points P

    # Declare parameters
    W = solver.parameter(dimension, nbr_wps)       # position waypoints
    Wvel = solver.parameter(dimension, nbr_wps)    # velocity waypoints
    knot_wp = solver.parameter(nbr_wps,1)            # knot/timestamp of waypoints


    # Cost function: integral of the norm of derivative of position(velocity)
    objective = 0
    P1 = P @ traj.M[0]                          # 1st-derivative control points
    P2 = P @ traj.M[1]                          # 2nd-derivative control points

    for i in range(nbr_ctrl_pts + 1):
        for j in range(nbr_ctrl_pts + 1):
            f_lamb = lambda t, it=i, jt=j: traj.bs[traj.bs_index[-2]][it](t) * traj.bs[traj.bs_index[-2]][jt](t)
            buff_int = quad(f_lamb, min(traj.knot), max(traj.knot))[0]
            objective = objective + casadi.mtimes(casadi.transpose(casadi.mtimes(buff_int, P1[:, i])), P1[:, j])

    # Implementing constraints
    # for i in range(W.shape[1]):
    #     tmp_bs = np.zeros((len(bs_list[0]), 1))
    #     for j in range(len(bs_list[0])):
    #         tmp_bs[j] = bs_list[0][j](waypoint_time_stamps[i])
    #     # Mathematically, mtimes(P, tmp_bs) = P * tmp_bs
    #     solver.subject_to(casadi.mtimes(P, tmp_bs) == W[:, i])
    for i in range(nbr_wps):
        # tmp = np.zeros((n+1,1))
        tmp = casadi.GenMX_zeros((n+1,1))
        for j in range(n+1):
            # np.squeeze( traj.bs[traj.bs_index[-1]][j](knot_wp[i]).full() )
            # print(np.size(traj.bs[traj.bs_index[-1]][j](knot_wp[i])))
            tmp[j] = traj.bs[traj.bs_index[-1]][j](knot_wp[i])
            # tmp[j] = np.squeeze( traj.bs[traj.bs_index[-1]][j](knot_wp[i]) )
            if j==1:
                tmp[0] = 2*tmp[0]
            elif j==n:
                tmp[-1] = 2*tmp[-1]
            # tmp[j,0] = 2*tmp[j,0]         # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
            # tmp[j,-1] = 2*tmp[j,-1]
            # tmp[j] = np.squeeze( traj.bs[traj.bs_index[-1]][j](knot_wp[i]).full() )
            # print(type(( traj.bs[traj.bs_index[-1]][j](knot_wp[i]) )))
            # tmp[j] = np.squeeze( traj.bs[traj.bs_index[-1]][j](knot_wp[i]) )

        # tmp1 = np.zeros((n+2,1))
        tmp1 = casadi.GenMX_zeros((n+2,1))
        for j in range(n+2):
            tmp1[j] = traj.bs[traj.bs_index[-2]][j](knot_wp[i])
            if j==1:
                tmp1[0] = 2*tmp1[0]
            elif j==n:
                tmp1[-1] = 2*tmp1[-1]
            # tmp1[j][0] = 2*tmp1[j,0]         # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
            # tmp1[j][-1] = 2*tmp1[j,-1]
            # tmp1[j] = np.squeeze( traj.bs[traj.bs_index[-2]][j](knot_wp[i]).full() )
            # tmp1[j] = np.squeeze( traj.bs[traj.bs_index[-2]][j](knot_wp[i]) )

        # tmp2 = np.zeros((n+3,1))
        tmp2 = casadi.GenMX_zeros((n+3,1))
        for j in range(n+3):
            tmp2[j,:] = traj.bs[traj.bs_index[-3]][j](knot_wp[i])
            if j==1:
                tmp2[0] = 2*tmp2[0]
            elif j==n:
                tmp2[-1] = 2*tmp2[-1]
            # tmp2[j,0] = 2*tmp2[j,0]         # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
            # tmp2[j,-1] = 2*tmp2[j,-1]
            # tmp2[j] = np.squeeze( traj.bs[traj.bs_index[-3]][j](knot_wp[i]).full() )
            # tmp2[j] = np.squeeze( traj.bs[traj.bs_index[-3]][j](knot_wp[i]) )


        solver.subject_to( P @ tmp == W[:,i])
        solver.subject_to( P1 @ tmp1 == Wvel[:,i])
        if i==1:
            solver.subject_to( P2 @ tmp2 == 0)
        # solver.subject_to( P2 @ tmp2 == 0)
    
    solver.minimize(objective)
    solver_options = {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': 0}
    solver.solver('ipopt', solver_options)

    solver_vars = {}
    solver_vars['P'] = P
    solver_vars['W'] = W
    solver_vars['Wvel'] = Wvel
    solver_vars['knot_wp'] = knot_wp
    solver_vars['tmp'] = tmp

    return solver, solver_vars

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.join(current_dir,'..')
sys.path.append(project_folder)

# set print options for numpy
np.set_printoptions(precision=5, suppress=True)

if __name__=="__main__":

    
    from traj_params import *
    from Bspline.bspline_casadi import *
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
    height = 0.7

    ### Leader's reference ###
    # initial point
    pinit_leader = np.array([-1.0, -1.5, height])
    pinit_leader = pinit_leader.reshape(-1,1)
    print(pinit_leader)
    vinit_leader = np.zeros((3,1))

    # intermediate points
    pinter_leader = np.array([[0.5, 0, height],
                              [1.5, 0.5, height],
                              [1.5, -1.0, height]]).transpose()
    print(pinter_leader)

    vinter_leader = np.zeros((3,3))
    print(vinter_leader)
    vinter_leader[1,0] = 0.01
    vinter_leader[1,1] = -0.1
    vinter_leader[0,2] = -0.1
    # vinter_leader[1,0] = 0.1
    print(vinter_leader)

    # final point
    pf_leader = np.array([0.5, 0,  height])
    pf_leader = pf_leader.reshape(-1,1)
    print(pf_leader)
    vf_leader = np.zeros((3,1))

    W_leader = np.hstack((pinit_leader,pinter_leader,pf_leader))
    print(W_leader)
    Wvel_leader = np.hstack((vinit_leader,vinter_leader,vf_leader))
    print(Wvel_leader)

    nbr_wps = np.size(W_leader, 1)
    print(nbr_wps)
    knot_wp = np.linspace(0, traj.Tf, nbr_wps)
    
    [solver, solver_vars] = get_solver_traj_waypoints(traj=traj, dimension=dimension, nbr_wps=nbr_wps)

    solver.set_value(solver_vars['W'], W_leader)
    solver.set_value(solver_vars['Wvel'], Wvel_leader)
    solver.set_value(solver_vars['knot_wp'], knot_wp)
    print(knot_wp)
    
    from time import time
    tic = time()
    sol = solver.solve()  # Solve for the control points
    toc = time()
    Elapsed_time = toc - tic
    print('Elapsed time for solving: ', Elapsed_time, '[second]')

    P = sol.value(solver_vars['P'])
    print(P)



    ctrl_pts = np.asarray(P)
    ctrl_pts_1 = ctrl_pts @ traj.M[0]
    ctrl_pts_2 = ctrl_pts @ traj.M[1]
    print(ctrl_pts)

    
    traj_tt = ctrl_pts @ traj.bs_eval        # trajectory
    vel_tt = ctrl_pts_1 @ traj.bs_eval1     # velocity - 1st derivative
    accel_tt = ctrl_pts_2 @ traj.bs_eval2   # acceleration - 2nd derivative

    ### External agent's reference ###
    # initial point
    pinit_ex = np.array([-1.5, 1.5, height])
    pinit_ex = pinit_ex.reshape(-1,1)
    print(pinit_leader)
    vinit_ex = np.zeros((3,1))

    # intermediate points
    pinter_ex = np.array([[0, 0.5, height],
                        [1.0, 1.0, height],
                        [0, 1.5, height]]).transpose()
    print(pinter_ex)

    vinter_ex = np.zeros((3,3))
    print(vinter_ex)
    vinter_ex[0,0] = 0.01
    vinter_ex[1,1] = 0.1
    vinter_ex[0,2] = -0.1
    # vinter_leader[1,0] = 0.1
    print(vinter_ex)

    # final point
    pf_ex = np.array([-1.5, 1.5,  height])
    pf_ex = pf_ex.reshape(-1,1)
    print(pf_ex)
    vf_ex = np.zeros((3,1))

    W_ex = np.hstack((pinit_ex,pinter_ex,pf_ex))
    print(W_ex)
    Wvel_ex = np.hstack((vinit_ex,vinter_ex,vf_ex))
    print(Wvel_ex)

    solver.set_value(solver_vars['W'], W_ex)
    solver.set_value(solver_vars['Wvel'], Wvel_ex)
    solver.set_value(solver_vars['knot_wp'], knot_wp)

    tic = time()
    sol = solver.solve()  # Solve for the control points
    toc = time()
    Elapsed_time = toc - tic
    print('Elapsed time for solving: ', Elapsed_time, '[second]')

    P_ex = sol.value(solver_vars['P'])
    print(P)



    ctrl_pts_ex = np.asarray(P_ex)
    ctrl_pts_1_ex = ctrl_pts_ex @ traj.M[0]
    ctrl_pts_2_ex = ctrl_pts_ex @ traj.M[1]
    print(ctrl_pts)

    
    traj_tt_ex = ctrl_pts_ex @ traj.bs_eval        # trajectory
    vel_tt_ex = ctrl_pts_1_ex @ traj.bs_eval1     # velocity - 1st derivative
    accel_tt_ex = ctrl_pts_2_ex @ traj.bs_eval2   # acceleration - 2nd derivative

    ### Plot results ###
    view_basisfunc = 1  # plot the basis functions
    view_3Dtraj = 1     # plot 3D trajectory
    view_pos = 1        # plot the position
    view_vel = 0        # plot the velocity
    view_acc = 0        # plot the acceleration

    # Number of colors
    nbr_colors = nbr_ctrl_pts + 2

    import matplotlib.cm as cm
    # Generate a list of n colors from a colormap
    colors = [cm.tab10(i) for i in range(nbr_colors)]

    import matplotlib.pyplot as plt
    # Enable LaTeX rendering for tick labels
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if view_basisfunc:
        fig_basis = plt.figure("Basis functions")
        # Basis functions for the current degree of B-spline
        ax = fig_basis.add_subplot(3,1,1)
        ax.grid(True)
        bs_plot = {}
        for i in range(n+1):
            bs_plot[i], = ax.plot(traj.tt,traj.bs_eval[i])
            
            plt.setp(bs_plot[i], color=colors[i], linestyle='-', linewidth=2.5)
        # ax.set_ylabel(r"${B_{i,k}}(t)$", usetex=True, fontsize=12)
        ax.set_ylabel(r"${B_{i,k}}(t)$", fontsize=12)

        # Basis functions for the 1st-derivative of B-spline
        ax = fig_basis.add_subplot(3,1,2)
        ax.grid(True)
        bs_1_plot = {}
        for i in range(n+2):
            bs_1_plot[i], = ax.plot(traj.tt,traj.bs_eval1[i])
            
            plt.setp(bs_1_plot[i], color=colors[i], linestyle='-', linewidth=2.5)
        # ax.set_ylabel(r"${B_{i,k-1}}(t)$", usetex=True, fontsize=12)
        ax.set_ylabel(r"${B_{i,k-1}}(t)$", fontsize=12)

        # Basis functions for the 2nd-derivative of B-spline
        ax = fig_basis.add_subplot(3,1,3)
        ax.grid(True)
        bs_2_plot = {}
        for i in range(n+2):
            bs_2_plot[i], = ax.plot(traj.tt,traj.bs_eval2[i])
            
            plt.setp(bs_2_plot[i], color=colors[i], linestyle='-', linewidth=2.5)
        # ax.set_ylabel(r"${B_{i,k-2}}(t)$", usetex=True, fontsize=12)
        # ax.set_xlabel(r"Time $t(s)$", usetex=True, fontsize=12)
        ax.set_ylabel(r"${B_{i,k-2}}(t)$", fontsize=12)
        ax.set_xlabel(r"Time $t(s)$", fontsize=12)

    # 3D trajectory
    if view_3Dtraj:
        # Create a 3D plot
        fig_3D = plt.figure("3D Trajectory plot",)
        ax = fig_3D.add_subplot(111, projection='3d')
        traj_3D_plot, = ax.plot(traj_tt[0,:], traj_tt[1,:], traj_tt[2,:])

        plt.setp(traj_3D_plot, color='b', linestyle='-', linewidth=2.5)

        ## external plot #######
        traj_3D_plot_ex, = ax.plot(traj_tt_ex[0,:], traj_tt_ex[1,:], traj_tt_ex[2,:])
        
        plt.setp(traj_3D_plot_ex, color='c', linestyle='-', linewidth=2.5)
        ########################

        # Change azimuth and elevation view
        ax.view_init(elev=20, azim=-80)

        # Set labels and title
        # ax.set_xlabel(r"$x(m)$", usetex=True, fontsize=12)
        # ax.set_ylabel(r"$y(m)$", usetex=True, fontsize=12)
        # ax.set_zlabel(r"$z(m)$", usetex=True, fontsize=12)
        # ax.set_title(r"3D Trajectory", usetex=True, fontsize=14)    
        ax.set_xlabel(r"$x(m)$", fontsize=12)
        ax.set_ylabel(r"$y(m)$", fontsize=12)
        ax.set_zlabel(r"$z(m)$", fontsize=12)
        ax.set_title(r"3D Trajectory", fontsize=14)

    # Position plot
    if view_pos:
        fig_pos = plt.figure("Position plot",)
        dimension_traj = 3
        pos_labels = ['x','y','z']
        pos_plot = {}

        for i in range(dimension_traj):
            ax = fig_pos.add_subplot(dimension_traj,1,i+1)
            ax.grid(True)

            pos_plot[i], = ax.plot(traj.tt, traj_tt[i,:])

            plt.setp(pos_plot[i], color='m', linestyle='-', linewidth=3.5)
            if i==0:
                # ax.set_title(r"Positions on $x$, $y$ and $z$-axis", usetex=True, fontsize=14)
                ax.set_title(r"Positions on $x$, $y$ and $z$-axis", fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", fontsize=12)
            ax.set_ylabel(f"${pos_labels[i]}(t)$", fontsize=12)

    # Velocity plot
    if view_vel:
        fig_vel = plt.figure("Velocity plot",)
        dimension_traj = 3
        vel_labels = ['v_{x}','v_{y}','v_{z}']
        vel_plot = {}

        for i in range(dimension_traj):
            ax = fig_vel.add_subplot(dimension_traj,1,i+1)
            ax.grid(True)

            vel_plot[i], = ax.plot(traj.tt, vel_tt[i,:])

            plt.setp(vel_plot[i], color='m', linestyle='-', linewidth=3.5)
            if i==0:
                ax.set_title(r"Velocity on $x$, $y$ and $z$-axis", fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", fontsize=12)
            ax.set_ylabel(f"${vel_labels[i]}(t)$", fontsize=12)

    # Acceleration plot
    if view_acc:
        fig_acc = plt.figure("Acceleration plot",)
        dimension_traj = 3
        acc_labels = ['a_{x}','a_{y}','a_{z}']
        acc_plot = {}

        for i in range(dimension_traj):
            ax = fig_acc.add_subplot(dimension_traj,1,i+1)
            ax.grid(True)

            acc_plot[i], = ax.plot(traj.tt, accel_tt[i,:])

            plt.setp(acc_plot[i], color='m', linestyle='-', linewidth=3.5)
            if i==0:
                ax.set_title(r"Acceleration on $x$, $y$ and $z$-axis", fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", fontsize=12)
            ax.set_ylabel(f"${acc_labels[i]}(t)$", fontsize=12)

    plt.show()


