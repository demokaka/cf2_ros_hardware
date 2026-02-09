from bspline_casadi import *
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
if __name__=="__main__":
    nbr_ctrl_pts = 11       # number of control points of B-spline
    n = nbr_ctrl_pts - 1    # spline cardinality
    deg = 4                 # degree of spline
    Tf = 30                 # final time
    knot_endpoints = [0,Tf]

    [bs, knot, x] = bsplines_casadi(n, deg, knot_endpoints)
    [M, Sd] = bsplineConversionMatrices(n, deg, knot)

    bs_index = list(bs.keys())

    # print(bs)
    # print(bs_index)
    # print(knot)
    # print(M)
    
    ## example: calculate the trajectory, velocity and acceleration
    # control points
    Xi=[[0, 0, 0, -1.2187, -1.4288, -0.8787, 1.4708, 0.5615, 0, 0, 0],
        [0, 0, 0, 0.3802, 0.5887, 0.5888, 0.2743, -0.9708, 0, 0, 0],
        [0, 0, 0, 0.2649, 0.2032, 0.9542, 0.7308, 0.5755, 0, 0, 0]]

    h=0.1
    Nb=int(Tf/h)
    tt=np.linspace(0,Tf,Nb+1);  # timestamps

    bs_eval = np.zeros((n+1,len(tt)))   # evaluation of basis function matrix at given the timestamps
    bs_eval_1 = np.zeros((n+2,len(tt)))
    bs_eval_2 = np.zeros((n+3,len(tt)))

    for i in range(n+1):
        bs_eval[i] = np.squeeze( bs[bs_index[-1]][i](tt).full() ) # np.squeeze() method removes single-dimensional entries from the shape of an array.
        bs_eval[i][0] = 2*bs_eval[i][0]         # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
        bs_eval[i][-1] = 2*bs_eval[i][-1]

    print(tt)
    print(bs[bs_index[-1]][i](tt).full())
    print(knot)
    print(bs_eval)
    for i in range(n+2):
        bs_eval_1[i] = np.squeeze( bs[bs_index[-2]][i](tt).full() ) # np.squeeze() method removes single-dimensional entries from the shape of an array.
        bs_eval_1[i][0] = 2*bs_eval_1[i][0]     # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
        bs_eval_1[i][-1] = 2*bs_eval_1[i][-1]

    for i in range(n+3):
        bs_eval_2[i] = np.squeeze( bs[bs_index[-3]][i](tt).full() ) # np.squeeze() method removes single-dimensional entries from the shape of an array.
        bs_eval_2[i][0] = 2*bs_eval_2[i][0]     # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
        bs_eval_2[i][-1] = 2*bs_eval_2[i][-1]

    ctrl_pts = np.asarray(Xi)
    ctrl_pts_1 = ctrl_pts @ M[0]
    ctrl_pts_2 = ctrl_pts @ M[1]

    traj_tt = ctrl_pts @ bs_eval        # trajectory
    vel_tt = ctrl_pts_1 @ bs_eval_1     # velocity - 1st derivative
    accel_tt = ctrl_pts_2 @ bs_eval_2   # acceleration - 2nd derivative

    ### Plot results ###
    view_basisfunc = 1  # plot the basis functions
    view_3Dtraj = 1     # plot 3D trajectory
    view_pos = 1        # plot the position
    view_vel = 0        # plot the velocity
    view_acc = 0        # plot the acceleration

    # Number of colors
    nbr_colors = nbr_ctrl_pts + 2

    # Generate a list of n colors from a colormap
    colors = [cm.tab10(i) for i in range(nbr_colors)]

    # Enable LaTeX rendering for tick labels
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if view_basisfunc:
        fig_basis = plt.figure("Basis functions")
        # Basis functions for the current degree of B-spline
        ax = fig_basis.add_subplot(3,1,1)
        ax.grid(True)
        bs_plot = {}
        for i in range(n+1):
            bs_plot[i], = ax.plot(tt,bs_eval[i])
            
            plt.setp(bs_plot[i], color=colors[i], linestyle='-', linewidth=2.5)
        ax.set_ylabel(r"${B_{i,k}}(t)$", usetex=True, fontsize=12)

        # Basis functions for the 1st-derivative of B-spline
        ax = fig_basis.add_subplot(3,1,2)
        ax.grid(True)
        bs_1_plot = {}
        for i in range(n+2):
            bs_1_plot[i], = ax.plot(tt,bs_eval_1[i])
            
            plt.setp(bs_1_plot[i], color=colors[i], linestyle='-', linewidth=2.5)
        ax.set_ylabel(r"${B_{i,k-1}}(t)$", usetex=True, fontsize=12)

        # Basis functions for the 2nd-derivative of B-spline
        ax = fig_basis.add_subplot(3,1,3)
        ax.grid(True)
        bs_2_plot = {}
        for i in range(n+2):
            bs_2_plot[i], = ax.plot(tt,bs_eval_2[i])
            
            plt.setp(bs_2_plot[i], color=colors[i], linestyle='-', linewidth=2.5)
        ax.set_ylabel(r"${B_{i,k-2}}(t)$", usetex=True, fontsize=12)
        ax.set_xlabel(r"Time $t(s)$", usetex=True, fontsize=12)
        

    # 3D trajectory
    if view_3Dtraj:
        # Create a 3D plot
        fig_3D = plt.figure("3D Trajectory plot",)
        ax = fig_3D.add_subplot(111, projection='3d')
        traj_3D_plot, = ax.plot(traj_tt[0,:], traj_tt[1,:], traj_tt[2,:])

        plt.setp(traj_3D_plot, color='b', linestyle='-', linewidth=2.5)
        # Change azimuth and elevation view
        ax.view_init(elev=20, azim=-80)

        # Set labels and title
        ax.set_xlabel(r"$x(m)$", usetex=True, fontsize=12)
        ax.set_ylabel(r"$y(m)$", usetex=True, fontsize=12)
        ax.set_zlabel(r"$z(m)$", usetex=True, fontsize=12)
        ax.set_title(r"3D Trajectory", usetex=True, fontsize=14)    

    # Position plot
    if view_pos:
        fig_pos = plt.figure("Position plot",)
        dimension_traj = 3
        pos_labels = ['x','y','z']
        pos_plot = {}

        for i in range(dimension_traj):
            ax = fig_pos.add_subplot(dimension_traj,1,i+1)
            ax.grid(True)

            pos_plot[i], = ax.plot(tt, traj_tt[i,:])

            plt.setp(pos_plot[i], color='m', linestyle='-', linewidth=3.5)
            if i==0:
                ax.set_title(r"Positions on $x$, $y$ and $z$-axis", usetex=True, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=True, fontsize=12)
            ax.set_ylabel(f"${pos_labels[i]}(t)$", usetex=True, fontsize=12)

    # Velocity plot
    if view_vel:
        fig_vel = plt.figure("Velocity plot",)
        dimension_traj = 3
        vel_labels = ['v_{x}','v_{y}','v_{z}']
        vel_plot = {}

        for i in range(dimension_traj):
            ax = fig_vel.add_subplot(dimension_traj,1,i+1)
            ax.grid(True)

            vel_plot[i], = ax.plot(tt, vel_tt[i,:])

            plt.setp(vel_plot[i], color='m', linestyle='-', linewidth=3.5)
            if i==0:
                ax.set_title(r"Velocity on $x$, $y$ and $z$-axis", usetex=True, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=True, fontsize=12)
            ax.set_ylabel(f"${vel_labels[i]}(t)$", usetex=True, fontsize=12)
    
    # Acceleration plot
    if view_acc:
        fig_acc = plt.figure("Acceleration plot",)
        dimension_traj = 3
        acc_labels = ['a_{x}','a_{y}','a_{z}']
        acc_plot = {}

        for i in range(dimension_traj):
            ax = fig_acc.add_subplot(dimension_traj,1,i+1)
            ax.grid(True)

            acc_plot[i], = ax.plot(tt, accel_tt[i,:])

            plt.setp(acc_plot[i], color='m', linestyle='-', linewidth=3.5)
            if i==0:
                ax.set_title(r"Acceleration on $x$, $y$ and $z$-axis", usetex=True, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=True, fontsize=12)
            ax.set_ylabel(f"${acc_labels[i]}(t)$", usetex=True, fontsize=12)

    plt.show()


    

    

