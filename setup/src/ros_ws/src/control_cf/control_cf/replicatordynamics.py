import numpy as np

def replicatordynamics(t,p,A,ref,gap):
    """
        Input: 
            -   t       :       time
            -   p       :       current strategic distribution
            -   A       :       adjacency matrix
            -   ref     :       reference trajectory of leader
            -   gap     :       relative position of follower w.r.t leader in a population
        Output:
            -   dpdt    :       replicator dynamics
    """
    
    p = p.reshape(-1, 1)
    F = gap.reshape(-1, 1) - p
    # F[-1] = -ref
    F[0] = -ref

    dpdt = 1.0 * np.diagflat(p) @ (np.diagflat(F) @ A @ p - A @ np.diagflat(F) @ p)

    return dpdt.flatten()