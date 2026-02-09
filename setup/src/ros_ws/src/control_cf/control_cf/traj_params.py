import numpy as np

class BsplineTrajParams:
    def __init__(self,  nbr_ctrl_pts: int,  # number of control points 
                        deg: int,           # degree of Bspline
                        bs,                 # set of basis functions
                        bs_index,           # (with bs), index to take out the corresponding basis function
                        M,                  # set of conversion matrices
                        knot,               # knot vector
                        Tf=None,            # total horizon
                        h=None,             # sampling time
                        ):  
        self.nbr_ctrl_pts = nbr_ctrl_pts    
        self.n = self.nbr_ctrl_pts - 1
        self.deg = deg
        self.bs = bs
        self.bs_index = bs_index
        self.M = M
        self.knot = knot

        self.knot_endpoints = np.array([min(knot),max(knot)])
        self.Tf= Tf
        self.h = h
        if self.Tf is not None and self.h is not None:
            # self.Nb = int(self.Tf/self.h)
            self.Nb = int(self.Tf/self.h)+1
            self.tt = np.linspace(0, self.Tf, self.Nb)

            bs_eval = np.zeros((self.n+1, len(self.tt)))
            bs_eval1 = np.zeros((self.n+2, len(self.tt)))
            bs_eval2 = np.zeros((self.n+3, len(self.tt)))

            for i in range(self.n +1):
                bs_eval[i] = np.squeeze( bs[bs_index[-1]][i](self.tt).full() ) # np.squeeze() method removes single-dimensional entries from the shape of an array.
                bs_eval[i][0] = 2*bs_eval[i][0]         # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
                bs_eval[i][-1] = 2*bs_eval[i][-1]
            
            for i in range(self.n +2):
                bs_eval1[i] = np.squeeze( bs[bs_index[-2]][i](self.tt).full() ) # np.squeeze() method removes single-dimensional entries from the shape of an array.
                bs_eval1[i][0] = 2*bs_eval1[i][0]         # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
                bs_eval1[i][-1] = 2*bs_eval1[i][-1]

            for i in range(self.n +3):
                bs_eval2[i] = np.squeeze( bs[bs_index[-3]][i](self.tt).full() ) # np.squeeze() method removes single-dimensional entries from the shape of an array.
                bs_eval2[i][0] = 2*bs_eval2[i][0]         # these 2 lines are essential due to the fact that the heaviside function has the value 1/2 at the extreme points
                bs_eval2[i][-1] = 2*bs_eval2[i][-1]
            
            self.bs_eval = bs_eval
            self.bs_eval1 = bs_eval1
            self.bs_eval2 = bs_eval2

        else:
            self.Nb = None
            self.tt = None
            self.bs_eval = None
            self.bs_eval1 = None
            self.bs_eval2 = None