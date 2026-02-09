import numpy as np

"""
    Single agent
"""
def concat_ref(ref_a,ref_b):

    ref = np.vstack((ref_a,ref_b))
    
    return ref

"""
    Multiple agents
"""
def concatenate_ref(ref_a,ref_b):               
    nbr_agents = len(ref_a)
    ref = {}
    for i in range(nbr_agents):
        ref[i] = np.vstack((ref_a[i],ref_b[i]))
    
    return ref

"""
    Single agent
"""
def concat_element_ref(ref_1,ref_2):
    ref = np.hstack((ref_1,ref_2))
    return ref

"""
    Multiple agents
"""
def concatenate_element_ref(ref_1,ref_2):
    nbr_agents = len(ref_1)
    ref = {}
    for i in range(nbr_agents):
        ref[i] = np.hstack((ref_1[i],ref_2[i]))
    return ref

"""
    Single agent
"""
def from_setpoints_to_ref(x,y,z,Ts,Tsim):
    # nbr_agents = np.size(x,0)
    knot = [0,Tsim]
    ref = {}
    # for i in range(nbr_agents):
    W = np.zeros((3,x.shape[0]))
    # print(W)
    W[0,:] = x
    W[1,:] = y
    W[2,:] = z
    
    k_pass = 1
    ref_tmp = np.empty((0, 3))
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(W[:, i_tmp])
        while Ts * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1

    ref = ref_tmp
    return ref

"""
    Multiple agents
"""
def from_setpoints_to_reference(x,y,z,Ts,Tsim):
    nbr_agents = np.size(x,0)
    knot = [0,Tsim]
    W = {}
    ref = {}
    for i in range(nbr_agents):
        W[i] = np.zeros((3,x[i,:].shape[0]))
        # print(W)
        W[i][0,:] = x[i,:]
        W[i][1,:] = y[i,:]
        W[i][2,:] = z[i,:]
    
        k_pass = 1
        ref_tmp = np.empty((0, 3))
        waypoint_time_stamps = np.linspace(min(knot), max(knot), W[i].shape[1] + 1)
        for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
            cur = np.array(W[i][:, i_tmp])
            while Ts * k_pass <= waypoint_time_stamps[i_tmp + 1]:
                ref_tmp = np.vstack((ref_tmp, cur))
                k_pass = k_pass + 1

        ref[i] = ref_tmp
    return ref

"""
    Single agent:
        - Concatenate 2 reference trajectories, including 
          'trajectory':(position,velocity), 'v_ref':(acceleration), 'time_step':(tt),... not only reference
"""
def concat_reftraj(ref_a,ref_b):
    Nb_a = np.size(ref_a['time_step'],0)
    Nb_b = np.size(ref_b['time_step'],0)
    # Nb_landing = np.size(ref_landing[0]['trajectory'], 0)
    ref_full = {}

    # traj_tmp = np.hstack( (ref_a['trajectory'], ref_b['trajectory']) )
    # traj_tmp = np.hstack( (traj_tmp, kron( ref['trajectory'][:,0],  np.ones((1,Nb_takeoff)) )) )
    # traj_tmp = np.hstack( (traj_tmp, ref_landing[3]['trajectory'].T) )

    ref_full['trajectory'] = np.hstack( (ref_a['trajectory'], ref_b['trajectory']) )
    ref_full['v_ref'] = np.hstack( (ref_a['v_ref'], ref_b['v_ref']) )
    ref_full['time_step'] = np.linspace(0,ref_a['time_step'][-1]+ref_b['time_step'][-1]+ref_a['time_step'][1],Nb_a+Nb_b)
    ref_full['phi'] = np.hstack( (ref_a['phi'], ref_b['phi']) )
    ref_full['theta'] = np.hstack( (ref_a['theta'], ref_b['theta']) )
    ref_full['thrust'] = np.hstack( (ref_a['thrust'], ref_b['thrust']) )
    ref_full['Nsim'] = Nb_a + Nb_b
    return ref_full





