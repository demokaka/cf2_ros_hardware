import numpy as np
import scipy as sp
import yaml
import osqp
from scipy import sparse


class CBFQPSolver():
    def __init__(self, v_ref0, x_ref0, x0, a1, a2, dc):
        self.prob = self.get_CBFQP_solver(v_ref0, x_ref0, x0, a1, a2, dc)
        self.a1 = a1
        self.a2 = a2
        self.dc = dc


    def get_CBFQP_solver(self, v_ref,x_ref,x, a1, a2, dc):

        # Generate problem data
        delta_x = x-x_ref
        # a = sp.linalg.block_diag(a2*np.eye(3),a1*np.eye(3))
        # print(a)
        common_part = v_ref - a2 * delta_x[0:3] - a1 * delta_x[3:6]
        # print(v_ref)
        # print(delta_x)
        # print(a2 * delta_x[0:3])
        # print(a1 * delta_x[3:6])
        delta = np.array([dc,dc,dc])


        # Create an OSQP object
        prob = osqp.OSQP()

        # OSQP data
        # P = sparse.eye(3)
        P = sparse.csc_matrix([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
        q = -v_ref
        # print(q)
        
        # A = sparse.eye(3)
        A = sparse.csc_matrix([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
        
        l = -common_part - a2 * delta
        u = common_part + a2 * delta


        # Setup workspace
        prob.setup(P, q, A, l, u,verbose=False, alpha=1,)

        return prob
    
    def update(self,v_ref,x_ref,x):
        delta_x = x-x_ref
        # a = sp.linalg.block_diag(a2*np.eye(3),a1*np.eye(3))
        # print(a)
        common_part = v_ref - self.a2 * delta_x[0:3] - self.a1 * delta_x[3:6]
        delta = np.array([self.dc,self.dc,self.dc])

        q_new = -v_ref
        l_new = common_part - self.a2 * delta
        u_new = common_part + self.a2 * delta

        self.prob.update(q=q_new,l=l_new,u=u_new)

# with open('Config_Crazyflie_2.yaml') as f:
#     system_parameters = yaml.load(f,Loader=yaml.FullLoader)

# qtm_ip = system_parameters['qtm_ip']
# Ts = system_parameters['Ts']
# Tsim = system_parameters['Tsim']
# m = system_parameters['mass']
# uris = system_parameters['uris']
# drone_bodies = system_parameters['drone_bodies']

# if __name__=="__main__":
#     ### Import trajectory for testing ###
#     import Trajectory_generation as trajgen
#     import get_solver_cmpc as cmpc
#     dc=0.1
#     a1=6
#     a2=8

#     ref = {}
#     vref = {}

#     full_ref1 = trajgen.get_ref_setpoints_Khanh(psi=0,Tsim=Tsim,dt=Ts,agent=1)
#     full_ref2 = trajgen.get_ref_setpoints_Khanh(psi=0,Tsim=Tsim,dt=Ts,agent=2)
#     full_ref3 = trajgen.get_ref_setpoints_Khanh(psi=0,Tsim=Tsim,dt=Ts,agent=3)

#     ref = {uris[0]: full_ref1["trajectory"],
#         uris[1]: full_ref2["trajectory"],
#         uris[2]: full_ref3["trajectory"]} 

#     vref = {uris[0]: full_ref1["v_ref"],
#             uris[1]: full_ref2["v_ref"],
#             uris[2]: full_ref3["v_ref"]}
    
#     common_plant,common_controller = cmpc.load_constant_parameters(Ts=Ts)
#     simulator = {}
#     simulator['Nsim'] = np.size(ref[uris[0]],0)
#     simulator['na'] = len(drone_bodies)

#     simulator['u_sim'] = np.zeros((common_plant['du'],simulator['Nsim'],simulator['na']))
#     simulator['x_sim'] = np.zeros((common_plant['dx'],simulator['Nsim']+1,simulator['na']))
    
#     solver = {}
#     solver[drone_bodies[0]] = CBFQPSolver(v_ref0=vref[uris[0]][0,:],
#                                              x_ref0=ref[uris[0]][0,:],
#                                              x0=simulator['x_sim'][:,0,0],
#                                              a1=a1, a2=a2,dc=dc)
    
#     res = solver[drone_bodies[0]].prob.solve()
#     print(res)
#     print(res.x)
    

#     solver[drone_bodies[0]].update(v_ref=vref[uris[0]][10,:],
#                      x_ref=ref[uris[0]][10,:],
#                     x=simulator['x_sim'][:,10,0],)
    
#     res = solver[drone_bodies[0]].prob.solve()
#     print(res)
#     print(res.x)
    

    

