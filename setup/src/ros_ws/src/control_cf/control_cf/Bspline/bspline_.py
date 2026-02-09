import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# def B(x, k, i, t):
#    if k == 0:
#       return 1.0 if t[i] <= x < t[i+1] else 0.0
#    if t[i+k] == t[i]:
#       c1 = 0.0
#    else:
#       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
#    if t[i+k+1] == t[i+1]:
#       c2 = 0.0
#    else:
#       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
#    return c1 + c2

# def B(x, k, i, t):
#     x = np.asarray(x)   # x is an array
#     if k == 0:
#         # return np.where(1.0 if t[i] <= x < t[i+1] else 0.0)
#         return np.where((t[i] <= x) & (x< t[i+1]), 1.0, 0.0)
#     if t[i+k] == t[i]:
#         # c1 = 0.0
#         c1 = 0.0 + np.zeros_like(x)
#     else:
#         c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
#     if t[i+k+1] == t[i+1]:
#         # c2 = 0.0
#         c2 = 0.0 + np.zeros_like(x)
#     else:
#         c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
#     return c1 + c2

# def bspline(x, t, c, k):
#    n = len(t) - k - 1
#    assert (n >= k+1) and (len(c) >= n)
#    return sum(c[i] * B(x, k, i, t) for i in range(n))

# def bsplines_(n, deg, knot = [0, 1]):
#     # knot calculation
#     if (knot.__len__() != n+1+deg):     # if the knot is not yet set, choose uniform clamped knot vector             
#         knot = np.concatenate((np.ones(deg - 1) * min(knot),
#                                np.linspace(min(knot), max(knot), n - deg + 3),
#                                np.resize(np.ones((1, deg - 1), dtype=int), deg - 1) * max(knot)))

#     bs = []

#     for i in range(deg + 1):
#         list_tmp = []
#         for j in range(i + n+1):
#             coeff_bspline = [0] * (n+1 + i)
#             coeff_bspline[j] = 1.0
#             basis_spline_tmp = lambda t: bspline(t, knot, coeff_bspline, deg - i)
#             list_tmp.append(basis_spline_tmp)
#         bs.append(list_tmp)
#     return bs, knot

def bsplines(n, deg, knot = [0, 1]):
    # knot calculation
    if (knot.__len__() != n+1+deg):     # if the knot is not yet set, choose uniform clamped knot vector             
        knot = np.concatenate((np.ones(deg - 1) * min(knot),
                               np.linspace(min(knot), max(knot), n - deg + 3),
                               np.resize(np.ones((1, deg - 1), dtype=int), deg - 1) * max(knot)))
    
    # generate basis functions using scipy.interpolate.Bspline(t,c,k)
    # where:
    # - t   :   knot
    # - c   :   coefficient, corresponding to the control points
    # - k   :   degree of Bspline
    # Here, 
    # t is the knot vector
    # c is chosen as following:
    # - for 0-derivative: there are (nbr_ctrl_pts) basis functions
    #   bs[0][0] --> bs[0][n] === [1_0, 0_1,..., 0_n], [0_0, 1_1,..., 0_n], ... [0_0, 0_1,..., 1_n]
    # - for 1st-derivative: there are (nbr_ctrl_pts+1) basis functions    
    #   bs[1][0] --> bs[0][n+1] === [1_0, 0_1,..., 0_(n+1)], [0_0, 1_1,..., 0_(n+1)], ... [0_0, 0_1,..., 1_(n+1)]
    # ...
    # - for k-derivative: there are (nbr_ctrl_pts+k) basis functions    
    #   bs[k][0] --> bs[0][n+k] === [1_0, 0_1,..., 0_(n+k)], [0_0, 1_1,..., 0_(n+k)], ... [0_0, 0_1,..., 1_(n+k)]
    # k is degree of bspline === deg

    bs = []

    for i in range(deg + 1):
        list_tmp = []
        for j in range(i + n+1):
            coeff_bspline = [0] * (n+1 + i)
            coeff_bspline[j] = 1.0
            basis_spline_tmp = interpolate.BSpline(knot, coeff_bspline, deg - i)
            list_tmp.append(basis_spline_tmp)
        bs.append(list_tmp)
    return bs, knot

def bsplineConversionMatrices(n, d, knot):
    tmp = np.eye(n + 1)
    M = {}
    for r in range(d):
        M[r] = np.zeros((n + r + 1, n + r + 2))
        for i in range(n + r + 1):
            if knot[i + d - r - 1] == knot[i]:
                M[r][i, i] = 0
            else:
                M[r][i, i] = (d - r - 1) / (knot[i + d - r - 1] - knot[i])
            if knot[i + d - r] == knot[i + 1]:
                M[r][i, i + 1] = 0
            else:
                M[r][i, i + 1] = -(d - r - 1) / (knot[i + d - r] - knot[i + 1])
        tmp = tmp @ M[r]
        M[r] = tmp
    Sd = []
    return M, Sd





if __name__=="__main__":
    import matplotlib.cm as cm
    import latex 
    nbr_ctrl_pts = 11
    n = nbr_ctrl_pts - 1    # spline cardinality
    deg = 4                 # degree of spline
    Tf = 30                 # final time
    knot_endpoints = [0,Tf]

    [bs, knot] = bsplines(n, deg, knot_endpoints)
    [M, Sd] = bsplineConversionMatrices(n, deg, knot)

    print(knot)
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
        bs_eval[i] = np.array(bs[0][i](tt))
        # bs_eval[i] = np.squeeze( bs[0][i](tt).full() ) # np.squeeze() method removes single-dimensional entries from the shape of an array.
        # bs_eval[i][0] = 2*bs_eval[i][0]         # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
        # bs_eval[i][-1] = 2*bs_eval[i][-1]

    print(tt)
    print(bs[0][i](tt))
    print(bs_eval)

    for i in range(n+2):
        bs_eval_1[i] = np.array(bs[1][i](tt))
        # bs_eval_1[i] = np.squeeze( bs[1][i](tt).full() ) # np.squeeze() method removes single-dimensional entries from the shape of an array.
        # bs_eval_1[i][0] = 2*bs_eval_1[i][0]     # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
        # bs_eval_1[i][-1] = 2*bs_eval_1[i][-1]

    for i in range(n+3):
        bs_eval_2[i] = np.array(bs[2][i](tt))
        # bs_eval_2[i] = np.squeeze( bs[2][i](tt).full() ) # np.squeeze() method removes single-dimensional entries from the shape of an array.
        # bs_eval_2[i][0] = 2*bs_eval_2[i][0]     # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
        # bs_eval_2[i][-1] = 2*bs_eval_2[i][-1]


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
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

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

