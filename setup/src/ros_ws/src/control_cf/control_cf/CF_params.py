import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from control import dlqr

class CFParameters:
    def __init__(self,drone_address,body_name,mass):
        self.uri = drone_address
        self.name = body_name
        self.mass = mass
    
    def load_plant_parameters(self, Ts, phy_lim=None,):
        plant = {}
        plant['A'] = np.vstack( (np.hstack( (np.zeros((3,3)),np.eye(3)) ), np.hstack( (np.zeros((3,3)), np.zeros((3,3))) )) )
        plant['B'] =  np.hstack( (np.zeros((3,3)), np.eye(3)) )
        plant['C'] =  np.hstack( (np.eye(3),np.zeros((3,3))))  
        plant['D'] = np.zeros(3)
        plant['Ad'] = np.vstack( (np.hstack( (np.eye(3),Ts*np.eye(3)) ), np.hstack( (np.zeros((3,3)), np.eye(3)) )) )
        plant['Bd'] = np.vstack( (np.eye(3)*0.5*Ts**2, Ts*np.eye(3)) )
        plant['Cd'] = np.hstack( (np.eye(3),np.zeros((3,3)))) 
        plant['Dd'] = np.zeros((3,3))

        plant['Ts'] = Ts                                              # sampling time
        # dimension
        plant['dx'] = np.size(plant['Bd'],0)                        
        plant['du'] = np.size(plant['Bd'],1)
        plant['dy'] = np.size(plant['Cd'],0)
        if not phy_lim: # by default
            plant['amin'] = np.array([[-1],[-1],[-1.5]])*0.9              # m/s²
            plant['amax'] = np.array([[1],[1],[1.5]])*0.9                 
            plant['vmin'] = np.array([[-1],[-1],[-1]])*1.5              # m/s
            plant['vmax'] = np.array([[1],[1],[1]])*1.5
            plant['pmin'] = np.array([[-1.8],[-1.8],[0]])*1.0           # m
            plant['pmax'] = np.array([[1.8],[1.8],[1.8]])*1.0
        else:
            plant['amin'] = phy_lim['amin']
            plant['amax'] = phy_lim['amax']               
            plant['vmin'] = phy_lim['vmin']           
            plant['vmax'] = phy_lim['vmax']
            plant['pmin'] = phy_lim['pmin']
            plant['pmax'] = phy_lim['pmax']
        
        self.plant = plant
    
    def load_controller_parameters(self,Q=None,P=None,R=None,Npred=None):
        controller = {}
        if not (Q): # by default
            # controller['Q'] = block_diag(np.eye(3)*50,np.eye(3)*5)  # Q = blkdiag(Qp,Qv)
            #                                                             # Qp = qp*I(dp),Qv = qv*I(dv)
            controller['Q'] = block_diag(np.eye(3)*500,np.eye(3)*250)  # Q = blkdiag(Qp,Qv)
                                                                        # Qp = qp*I(dp),Qv = qv*I(dv)
        else:
            controller['Q'] = Q
        if not (P): # by default
            controller['P'] = block_diag(np.eye(3)*50,np.eye(3)*5)  # P = blkdiag(Pp,Pv)
                                                                        # Pp = pp*I(dp),Pv = pv*I(dv)
        else:
            controller['P'] = P
        if not (R): # by default
            controller['R'] = np.eye(self.plant['du'])*10                 # R = r*I(du)
        else:
            controller['R'] = R
        
        if not (Npred): # by default
            controller['Npred'] = 5
        else:
            controller['Npred'] = Npred
        
        controller['K_lqr'],_,_ = dlqr(self.plant['Ad'], self.plant['Bd'], controller['Q'], controller['R'])
        controller['K_lqr'] = 2.0 *  np.array([[2.5, 0, 0, 1.5, 0, 0],
                [0, 2.5, 0, 0, 1.5, 0],
                [0, 0, 2.5, 0, 0, 1.5]])
        self.controller = controller
    
    def advance(self,xk,uk):
        xk1 = self.plant['Ad'] @ xk + self.plant['Bd'] @ uk
        return xk1


if __name__=="__main__":
    cf_params = CFParameters("07F:01","DroneE1",35.5)
    cf_params.load_plant_parameters(0.1)
    cf_params.load_controller_parameters()
    print(cf_params.controller)