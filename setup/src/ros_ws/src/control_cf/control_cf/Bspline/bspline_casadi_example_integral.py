from bspline_casadi import *
import numpy as np
import matplotlib.cm as cm

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

    # print(tt)
    # print(bs[bs_index[-1]][i](tt).full())
    # print(knot)
    # print(bs_eval)
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

    from scipy.integrate import quad
    objective = 0
    for i in range(nbr_ctrl_pts + 1):
        for j in range(nbr_ctrl_pts + 1):
            f_lamb = lambda t, it=i, jt=j: bs[bs_index[-2]][it](t) * bs[bs_index[-2]][jt](t)
            buff_int = quad(f_lamb, min(knot), max(knot))[0]
            objective = objective + mtimes(transpose(mtimes(buff_int, ctrl_pts_1[:, i])), ctrl_pts_1[:, j])

    print(objective)