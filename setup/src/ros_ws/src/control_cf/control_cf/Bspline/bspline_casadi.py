from casadi import *
import scipy.io as sio
import numpy as np
# import matplotlib.pyplot as plt

def bsplines_casadi(n, deg, knot = [0, 1]):
    if (knot.__len__() != n+1+deg):
        knot = np.concatenate((np.ones(deg - 1) * min(knot),
                               np.linspace(min(knot), max(knot), n - deg + 3),
                               np.resize(np.ones((1, deg - 1), dtype=int), deg - 1) * max(knot)))
    m = knot.__len__() - 1

    x = SX.sym('x', 1)

    bs = {}
    l = {}
    # generate recursively the family of splines goind until order deg+1
    for i in range(m):
        l[i] = Function('f_' + str(0) + '_' + str(i), [x], [heaviside(x-knot[i])-heaviside(x-knot[i+1])])
    bs[0] = l
    # apply the recursive bpsline relation (eq. (2.1) at page 4)
    for k in range(1, deg):
        l = {}
        for i in range(m - k):
             l[i] = Function('f_' + str(k) + '_' + str(i), [x],
                [if_else(knot[i + k] - knot[i] != 0, bs[k - 1][i](x)*(x - knot[i])/(knot[i + k]-knot[i]), 0)
                + if_else(knot[i + k + 1] - knot[i + 1] != 0, bs[k - 1][i + 1](x) * (knot[i + k + 1] - x) / (knot[i + k + 1] - knot[i + 1]), 0)])
        bs[k] = l
    return bs, knot, x

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

