#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
import numpy as np
import yaml
import os
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from nav_msgs.msg import Odometry
from motion_capture_tracking_interfaces.msg import NamedPose, NamedPoseArray

from control_cf import control_packagecf as cfcontrol
# from control_cf import Trajectory_generation as trajgen
# from control_cf import generate_traj
from control_cf.pid_fl import *
# from control_cf.mpc_solvers import *
from control_cf.Functions import traj_gen_trc as trajgen
from control_cf.takeoff_landing import *

# plant, controller parameters
from control_cf.CF_params import *
from control_cf.get_solver_CBFQP import *

from functools import partial
import tf_transformations

from scipy.integrate import solve_ivp
from control_cf.replicatordynamics import *

# to save data from population dynamics
from control_cf.save_rds import *



QOSP = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT
)


class CrazyflieController(Node):
    def __init__(self):
        """ Initialize the Crazyflie controller node and set up all parameters, trajectories, and ROS 2 interfaces. """
        super().__init__('crazyflie_control_node')

        # Load parameters from YAML for the simulation
        config_file = os.path.join(get_package_share_directory('control_cf'), 'config', 'Config_Crazyflie_hardware.yaml')
        self.system_parameters = self.load_params(config_file)

        # controller_config = os.path.join(get_package_share_directory('control_cf'), 'config', 'Config_MPC.yaml')
        # solver_config = self.load_mpc_configuration(controller_config)

        self.no_drones = len(self.system_parameters['drone_bodies'])
        self.pose_subscriber = self.create_subscription(NamedPoseArray, f'/poses', self.poses_callback, QOSP, callback_group=ReentrantCallbackGroup())
        # self.odom_subscribers = {
        #     i: self.create_subscription(Odometry, f'/crazyflie_{i}/odom', partial(self.odom_callback, idx=i-1), 10)
        #     for i in range(1, self.no_drones + 1)
        # }
        self.state = dict.fromkeys(self.drone_bodies, np.empty((0, 13)))  # (x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz)
        self.current_step = 0

        self.controls = {}
        self.T_land = {}
        self.comp_times = []
        self.T_takeoff = 5
        self.T_hovering1 = 2
        self.T_hovering2 = 2
        self.T_landing = 5

        # Debugging Data
        self.output_data = dict.fromkeys(self.drone_bodies, np.empty((0, 30)))  # (self.state, T, roll, pitch, yaw, time, v, ref, vref)
        self.takeoff_data = dict.fromkeys(self.drone_bodies, np.empty((0, 30)))

        # replicator dynamics parameters
        self.rd_params = {}
        self.rd_params['ts'] = 0.1           # sampling time of replicator dynamics
        self.rd_params['nbr_steps'] = 30    # total rd steps
        # self.rd_params['rho'] = 0.85          # maximum distance of connectivity  (for 2 drones)
        self.rd_params['rho'] = 0.8         # maximum distance of connectivity (for 4 drones)

        # Mass of population
        self.Mass = 1e5

        # Formation design
        # g0 = 0.5
        self.g0 = 0.6
        # gap = np.zeros((3, nbr_agents))
        self.gap = np.zeros((3, 4))
        # self.gap[0,:] = [-self.g0, 0, -self.g0, 0]
        # self.gap[1,:] = [self.g0, self.g0, 0, 0]
        # self.gap[2,:] = [0, 0, 0, 0]
        self.gap[0,:] = [0, -self.g0, 0, -self.g0]
        self.gap[1,:] = [0, 0, self.g0, self.g0]
        self.gap[2,:] = [0, 0, 0, 0]

        self.gap_vel = np.zeros((3, 4))
        self.gap_acc = np.zeros((3, 4))

        # replicator dynamics design
        self.offset_pos = np.array([2.01, 2.01, 2.01]).reshape(-1, 1)
        self.rd_params['offset_pos'] = self.offset_pos
        self.ref, self.vref = self.load_trajectories()
        self.refto, self.vrefto = self.load_takeoff_traj()
        self.ref_land, self.vref_land = self.load_landing_point()

        # Define Controllers
        self.Kf, self.Klqi = self.set_controller()
        self.Kp = np.diag([2, 2, 2])
        self.Kd = np.diag([6, 6, 6])
        self.Ki = np.diag([1.5, 1.5, 1.5])

        # self.solver_data = setup_solver(solver_config)

        self.Nb = len(self.ref[self.uris[0]][0,:])
        self.Tf = self.Tsim

        ### Control parameters
        # plant and controller parameters 
        self.cf_params = {}
        for i in range(len(self.drone_bodies)):
            self.cf_params[self.drone_bodies[i]] = CFParameters(self.uris[i], self.drone_bodies[i], self.m[i])
            self.cf_params[self.drone_bodies[i]].load_plant_parameters(self.Ts)
            self.cf_params[self.drone_bodies[i]].load_controller_parameters()

        # CBF-QP solver initialization
        self.solver = {}
        self.dc = 0.1
        self.a1 = 6
        # self.a1 = 16
        self.a2 = 8
        # self.a2 = 18
        for i in range(len(self.drone_bodies)):
            self.solver[self.drone_bodies[i]] = CBFQPSolver(v_ref0=self.vref[self.uris[0]][:,0], x_ref0=self.ref[self.uris[0]][:,0],
                                                    x0=self.ref[self.uris[0]][:,0],
                                                    a1=self.a1, a2=self.a2, dc=self.dc)

        ### Simulator for simulation loop ###
        self.simulator = {}
        self.simulator['x_sim'] = np.zeros((6, self.no_drones, self.Nb+1))
        self.simulator['u_sim'] = np.zeros((3, self.no_drones, self.Nb))

        self.simulator['dist_ij'] = np.zeros((self.no_drones, self.no_drones, self.Nb))
        self.simulator['A_sim'] = np.zeros((self.no_drones, self.no_drones, self.Nb))
        self.simulator['F_sim'] = np.zeros((self.no_drones, 3, self.rd_params['nbr_steps'], self.Nb))
        self.simulator['p_sim'] = np.zeros((3, self.no_drones, self.rd_params['nbr_steps']+1, self.Nb))
        self.simulator['dp_sim'] = np.zeros((3, self.no_drones, self.rd_params['nbr_steps'], self.Nb))

        self.simulator['xref_sim'] = np.zeros((6, self.no_drones, self.Nb))

        self.start_swarm()

        self.flight_phase = "takeoff"
        self.pos_int = {i: np.zeros((3,1)) for i, _ in enumerate(self.drone_bodies)}

        self.roll_int = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.pitch_int = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.thrust_pwm_int = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.roll_p = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.pitch_p = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.thrust_p = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        # self.Thrust_pwm_eq = {i: int(cfcontrol.Thrust_to_PWM_modified(1.0, m=self.m[i])) for i, _ in enumerate(self.drone_bodies)}
        self.Thrust_pwm_eq = {i: int(cfcontrol.Thrust_to_PWM_quad(1.0, m=self.m[i], PWM0=self.PWM0[i])) for i, _ in enumerate(self.drone_bodies)}

        self.timer = self.create_timer(self.Ts, self.control_callback, callback_group=ReentrantCallbackGroup())

        self.start_time = time.time()

    def arm(self, scf):
        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)

    def start_swarm(self):
        """ Initializes the swarm after QTM is ready. """
        try:
            cflib.crtp.init_drivers(enable_debug_driver=False)
            factory = CachedCfFactory(rw_cache='./cache')

            self.swarm = Swarm(self.uris, factory=factory)
            self.swarm.open_links()
            
            self.swarm.reset_estimators()
            self.get_logger().info("Estimators have been reset")

            self.swarm.parallel_safe(self.arm)
            
            self.swarm.parallel_safe(self.wait_for_param_download)
            self.swarm.sequential(self.unlock_safety_check)

            self.get_logger().info("Swarm initialization complete.")
        except Exception as e:
            self.get_logger().error(f"Error during swarm initialization: {e}")
            self.shutdown_swarm()

    def shutdown_swarm(self):
        """ Shuts down the swarm. """
        try:
            if self.swarm:
                self.swarm.parallel_safe(self.land)
                self.swarm.close_links()
        except Exception as e:
            self.get_logger().error(f"Error shutting down swarm: {e}")

    def odom_callback(self, msg: Odometry, idx: int) -> None:
        """Callback function to save the nav message of the drone's odometry (pose)."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        twist = msg.twist.twist
        orientation = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        # self.state[self.drone_bodies[idx]] = [p.x, p.y, p.z,
        #                                  twist.linear.x, twist.linear.y, twist.linear.z,
        #                                  orientation[0], orientation[1], orientation[2],
        #                                  twist.angular.x, twist.angular.y, twist.angular.z]
        self.state[self.drone_bodies[idx]] = np.array([p.x, p.y, p.z,
                                         twist.linear.x, twist.linear.y, twist.linear.z,
                                         q.x, q.y, q.z, q.w,
                                         twist.angular.x, twist.angular.y, twist.angular.z]).reshape(1, -1)

    def poses_callback(self, msg: NamedPoseArray):
        """ ROS 2 callback to process pose updates from the motion capture system. """
        for i, named_pose in enumerate(msg.poses):
            position = named_pose.pose.position
            orientation = named_pose.pose.orientation
            linear_velocity = named_pose.velocity.linear
            angular_velocity = named_pose.velocity.angular

            if named_pose.name in self.state:
                self.state[named_pose.name] = np.array([
                    position.x, position.y, position.z,
                    linear_velocity.x, linear_velocity.y, linear_velocity.z,
                    orientation.x, orientation.y, orientation.z, orientation.w,
                    angular_velocity.x, angular_velocity.y, angular_velocity.z
                ]).reshape(1, -1)

    def load_params(self, config_file):
        """ Load parameters from config file. """
        with open(config_file) as f:
            system_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.simid = system_parameters['simid']
        self.g = 9.81
        # self.qtm_ip = system_parameters['qtm_ip']
        self.Ts = system_parameters['Ts']
        self.Tsim = system_parameters['Tsim']
        self.m = system_parameters['mass']
        self.uris = system_parameters['uris']
        self.drone_bodies = system_parameters['drone_bodies']
        self.T_coeff = system_parameters['T_coeff']
        self.alpha = system_parameters['alpha']
        self.PWM0 = system_parameters['PWM0']
        self.v_land = 0.1  # m/s
        self.Tto = 10  # Takeoff time in seconds

        return system_parameters

    def load_mpc_configuration(self, config_file):
        """ Load MPC solver configuration and precomputed matrices from files. """
        with open(config_file) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        
        Upos_data = os.path.join(get_package_share_directory('control_cf'), 'control_cf', 'Upos.npy')
        # data_Upos = np.load('src/crazyflie_control/crazyflie_control/Upos.npy', allow_pickle=True).tolist()
        data_Upos = np.load(Upos_data, allow_pickle=True).tolist()
        Vc = {}
        Vc['A_vc'] = np.round(data_Upos['A'][0, 0], 5)  # round up
        Vc['b_vc'] = np.round(data_Upos['b'][0, 0], 5)
        solver_config = {'Na': len(self.drone_bodies), 'Npred': params['Npred'], 'Vc': Vc, 'Ts': self.Ts,
                         'Q': np.array(params['Q'], dtype=np.float32), 'R': np.array(params['R'], dtype=np.float32)}
        return solver_config

    def load_trajectories(self):
        """ Load flight reference trajectories for each drone. """
        # full_ref = trajgen.get_ref_setpoints(psi=0, Tsim=self.Tsim, dt=self.Ts, version=0)
        # full_ref = generate_traj.get_ref(0, 30, self.Ts)
        ref, vref = {}, {}
        # full_ref = trajgen.get_ref_trc(30, self.Ts)
        P_leader, full_ref = trajgen.get_ref_trc_leader(height=0.8)
        P_ex, full_ref_ex = trajgen.get_ref_trc_ex(height=0.8)

        ref[self.uris[0]] = full_ref["trajectory"]
        vref[self.uris[0]] = full_ref["v_ref"]

        ref_ex = full_ref_ex["trajectory"]
        vref_ex = full_ref_ex["v_ref"]
        if self.no_drones > 1:
            for i in range(1, self.no_drones):
                if i <= 2:
                    ref[self.uris[i]] = full_ref["trajectory"] + np.vstack((self.gap[:, [i]], np.zeros((3, 1))))
                    vref[self.uris[i]] = full_ref["v_ref"]
                else:
                    ref[self.uris[i]] = ref_ex 
                    vref[self.uris[i]] = vref_ex
        self.ref_full_rd = full_ref['trajectory'][0:3,:] + self.offset_pos

        # ref[self.uris[0]] = full_ref["trajectory"] + np.array([0.5, 0.0, 0.0, 0, 0, 0])
        # vref[self.uris[0]] = full_ref["v_ref"]

        # ref[self.uris[1]] = full_ref["trajectory"] + np.array([-0.5, -0.5, 0.0, 0, 0, 0])
        # vref[self.uris[1]] = full_ref["v_ref"]

        # ref[self.uris[2]] = full_ref["trajectory"] + np.array([-0.5, 0.5, 0.0, 0, 0, 0])
        # vref[self.uris[2]] = full_ref["v_ref"]

        # full_ref = trajgen.get_ref_setpoints(psi=0, Tsim=self.Tsim, dt=self.Ts, version=4)
        # ref[self.uris[1]] = full_ref["trajectory"] #+ np.array([0.5, -0.5, 0.2, 0, 0, 0])
        # vref[self.uris[1]] = full_ref["v_ref"]

        # full_ref = trajgen.get_ref_setpoints(psi=0, Tsim=30, dt=self.Ts, version=3)
        # ref[self.uris[2]] = full_ref["trajectory"]
        # vref[self.uris[2]] = full_ref["v_ref"]
        #### Leader's reference trajectory generation using B-splines ####
        
        return ref, vref

    def load_takeoff_traj(self):
        """ Generate reference trajectories for drone takeoff. """
        refTo, vrefTo = {}, {}
        # for i, uri in enumerate(self.uris):
        #     full_refTo = trajgen.get_ref_setpoints_takeoff(psi=0, Tto=self.Tto, dt=self.Ts, ref=self.ref[self.uris[i]])
        #     refTo[uri] = full_refTo["trajectory"]
        #     vrefTo[uri] = full_refTo["v_ref"]
        
        for i, uri in enumerate(self.uris):                
            full_refTo = generate_takeoff_traj(self.ref[uri][0,0],
                                                self.ref[uri][1,0],
                                                self.ref[uri][2,0],
                                                self.T_takeoff,
                                                self.T_hovering1,
                                                self.Ts)
            
            refTo[uri] = full_refTo["trajectory"]
            vrefTo[uri] = full_refTo["v_ref"]

        return refTo, vrefTo


    def load_landing_point(self):
        """ Define the reference landing point for each drone. """
        ref_land, vref_land = {}, {}
        # for i, uri in enumerate(self.uris):
        #     last_state_ref = self.ref[uri][-1]  # make the landing point be the last point of the reference
        #     ref_land[uri] = np.array([[last_state_ref[0]], [last_state_ref[1]], [0.05], [0], [0], [0]]).reshape(1, -1)
        #     vref_land[uri] = np.array([[0], [0], [0]]).reshape(1, -1)
        for i, uri in enumerate(self.uris):
            last_state_ref = self.ref[uri][:,0]  # make the landing point be the last point of the reference
            ref_land[uri] = np.array([[last_state_ref[0]], [last_state_ref[1]], [0.05], [0], [0], [0]]).reshape(1, -1)
            vref_land[uri] = np.array([[0], [0], [0]]).reshape(1, -1)

        return ref_land, vref_land

    def set_controller(self):
        """ Define the LQR controller. """
        Kf, Klqi = {}, {}

        for i, uri in enumerate(self.uris):
            Kf[uri] = -2.0 * np.array([[2.5, 0, 0, 1.5, 0, 0],
                                       [0, 2.5, 0, 0, 1.5, 0],
                                       [0, 0, 2.5, 0, 0, 1.5]])
            # Kf[uri] = -2.0 * np.array([[2.5, 0, 0, 2.0, 0, 0],
            #                            [0, 2.5, 0, 0, 2.0, 0],
            #                            [0, 0, 2.5, 0, 0, 2.0]])

        A = np.diag([6.19, 6.19, 6.19])
        B = np.diag([4.24, 4.24, 4.24])
        C = np.diag([-2.58, -2.58, -2.58])
        Ki = np.hstack((A, B, C))

        for drone in self.drone_bodies:
            Klqi[drone] = Ki * -1.0

        return Kf, Klqi

    def wait_for_param_download(self, scf: Swarm):
        """ Wait for Crazyflie parameter download to complete before flight. """
        while not scf.cf.param.is_updated:
            time.sleep(1.0)
        self.get_logger().info(f'Parameters downloaded for {scf.cf.link_uri}.\n')

    def unlock_safety_check(self, scf: Swarm):
        """ Unlock Crazyflie safety mode by sending a zero setpoint. """
        self.get_logger().info(f'Unlocking safety for {scf.cf.link_uri}.\n')
        scf.cf.commander.send_setpoint(0, 0, 0, 0)

    def apply_control(self, scf: Swarm, controller_cf):
        """ Send computed control commands to a Crazyflie drone. """
        scf.cf.commander.send_setpoint(*controller_cf)

    def land_grad(self, scf, currentThrust, currentZ):
        """ Perform gradual landing by decreasing thrust smoothly over time. """
        T_land = float(currentZ[0]) / self.v_land
        for i in np.arange(0, T_land, self.Ts):
            scf.cf.commander.send_setpoint(0, 0, 0, int(currentThrust * (1 - 0.1*i/T_land)))

    def land(self, scf):
        """ Stop all commands and hand over control to the high-level commander for landing. """
        scf.cf.commander.send_stop_setpoint()
        # Hand control over to the high level commander to avoid timeout and locking of the Crazyflie
        scf.cf.commander.send_notify_setpoint_stop()

    def control_callback(self):
        """ Periodic timer callback to execute takeoff, flight, and landing phases. """

        if self.flight_phase == "takeoff":
            tic = time.time()
            # cf_control, _ = self.control_loop(self.current_step, self.refto, self.vrefto)
            cf_control, _ = self.control_loop_lqr(self.current_step, self.refto, self.vrefto)
            toc = time.time()
            self.comp_times.append(toc-tic)

            self.get_logger().info(f"Takeoff phase, step {self.current_step}/{len(self.refto[self.uris[0]][0,:])}")  
            self.swarm.parallel_safe(self.apply_control, args_dict=cf_control)
            self.current_step += 1
            if self.current_step == len(self.refto[self.uris[0]][0,:]):
                self.flight_phase = "flight"
                self.current_step = 0  # Reset step for normal trajectory

        elif self.flight_phase == "flight":
            #################################################
            ###### Game theory ##############################
            #################################################
            for i, drone in enumerate(self.drone_bodies):
                self.simulator['x_sim'][:, i, self.current_step] = (self.state[drone][:, 0:6].T).reshape(1,-1)
            if self.no_drones > 1:                              # if there is only 1 agent => do not run game theory !!!
                for i in range(self.no_drones-1):
                    for j in range(i+1, self.no_drones):
                        dij = self.simulator['x_sim'][0:3, i, self.current_step] - self.simulator['x_sim'][0:3, j, self.current_step]

                        self.simulator['dist_ij'][i,j,self.current_step] = np.linalg.norm(dij)
                        if self.simulator['dist_ij'][i,j,self.current_step] <= self.rd_params['rho']:
                            self.simulator['A_sim'][i,j,self.current_step] = 1
                            self.simulator['A_sim'][j,i,self.current_step] = 1
                print(f"Adjacency matrix at step {self.current_step}:\n {self.simulator['A_sim'][:,:,self.current_step]}")

                for j in range(3):

                    # current population state
                    pk =  self.simulator['x_sim'][j, :, self.current_step] + self.offset_pos[j, :]

                    self.simulator['p_sim'][j, :, 0, self.current_step] = pk
                    # simulator['p_sim'][j, -1, 0, k] = nbr_agents * ref_rd[j, k+1] + sum(gap[j, :, cnt]) - sum(pk[0:(nbr_agents-1)])
                    self.simulator['p_sim'][j, 0, 0, self.current_step] = self.Mass - sum(pk[1:self.no_drones])

                    # ODE45 method
                    t_span = (0, self.rd_params['nbr_steps']*self.rd_params['ts'])
                    t_eval = np.linspace(0, self.rd_params['nbr_steps']*self.rd_params['ts'], self.rd_params['nbr_steps']+1)
                    p0 = self.simulator['p_sim'][j, :, 0, self.current_step]
                    # print(f'p0 = {p0}')
                    # problem = lambda t, p, A_=simulator['A_sim'][:,:,k],ref_=ref_rd[j,k+1],gap_=gap[j,:,k_s[k]]: replicatordynamics(t, p, A_, ref_, gap_)
                    # problem = lambda t, p: replicatordynamics(t, p, simulator['A_sim'][:,:,k], ref_rd[j,k+1], gap[j,:])
                    problem = lambda t, p: replicatordynamics(t, p, self.simulator['A_sim'][:,:,self.current_step], self.ref_full_rd[j,self.current_step], self.gap[j,0:self.no_drones])
                    # problem = lambda t, p: replicatordynamics(t, p, simulator['A_sim'][:,:,cnt], ref_full_rd[j,cnt+1], gap[j,[0,3]])
                    rtol = 1e-6             # reduce the error tolerance of the solver(its default value of 1e-3 (0.001))
                                            # either we can adjust absolute tolerance using atol
                    # ode_method = 'RK45'    # 'rk45' by default. Others: Radau
                    ode_method = 'Radau'
                    solution = solve_ivp(problem, t_span=t_span, y0=p0, t_eval=t_eval, rtol=rtol, method=ode_method)

                    t_cur, p_cur = np.asarray(solution.t), np.asarray(solution.y)
                    
                    # print(t_cur)
                    # print(p_cur)
                    for m in range(self.rd_params['nbr_steps']):

                        self.simulator['p_sim'][j, :, m+1, self.current_step] = p_cur[:, m]
                        self.simulator['F_sim'][:,j,m,self.current_step] = self.gap[j,0:self.no_drones].T - p_cur[:, m]
                        # self.simulator['F_sim'][:,j,m,cnt] = gap[j,[0,3]].T - p_cur[:, m]
                        # self.simulator['F_sim'][-1,j,m,cnt] = -ref_rd[j, k+1]
                        self.simulator['F_sim'][-1,j,m,self.current_step] = -self.ref_full_rd[j, self.current_step]
                        self.simulator['dp_sim'][j, :, m, self.current_step] = (np.diag(self.simulator['p_sim'][j, :, m, self.current_step]).T) @ (np.diag(self.simulator['F_sim'][:, j, m, self.current_step]) 
                                                                                                        @ self.simulator['A_sim'][:,:,self.current_step] @ self.simulator['p_sim'][j, :, m, self.current_step]
                                                                                                        - self.simulator['A_sim'][:,:,self.current_step] @ np.diag(self.simulator['F_sim'][:, j, m, self.current_step]) 
                                                                                                        @ self.simulator['p_sim'][j, :, m, self.current_step])

                    self.simulator['xref_sim'][j,:,self.current_step] = self.simulator['p_sim'][j, :, -1, self.current_step] - self.offset_pos[j,:]

                    # if sum(self.simulator['A_sim'][0,:,self.current_step])<1 or self.current_step<(Tf/4 + T_takeoff+ T_hovering1)/Ts:
                    #     # self.simulator['xref_sim'][j,1,k] = traj_tt_ex[j,k+1]
                    #     self.simulator['xref_sim'][j,-1,self.current_step] = ref_full_ex['trajectory'][j,self.current_step+1]
                    # # self.simulator['xref_sim'][j,-1,k] = ref['trajectory'][j,k+1]
                    # self.simulator['xref_sim'][j,0,self.current_step] = ref_full['trajectory'][j,self.current_step+1]
                    if sum(self.simulator['A_sim'][-1,:,self.current_step])<1 or self.current_step<(self.Tf/4 )/self.Ts:
                        # self.simulator['xref_sim'][j,1,k] = traj_tt_ex[j,k+1]
                        self.simulator['xref_sim'][j,-1,self.current_step] = self.ref[self.uris[-1]][j,self.current_step]
                    # self.simulator['xref_sim'][j,-1,k] = ref['trajectory'][j,k+1]
                    self.simulator['xref_sim'][j,0,self.current_step] = self.ref[self.uris[0]][j,self.current_step]
                    self.get_logger().info(f"Position component {j} at step {self.current_step}: {[self.simulator['xref_sim'][j,i,self.current_step] for i in range(self.no_drones)]}") 
                for i in range(self.no_drones):
                    if (sum(self.simulator['A_sim'][-1,:,self.current_step])<1 or self.current_step<(self.Tf/4)/self.Ts) and i==self.no_drones-1:
                        # self.simulator['xref_sim'][3:,i,k] = ref_ex['trajectory'][3:,k+1]
                        self.simulator['xref_sim'][3:,i,self.current_step] = self.ref[self.uris[-1]][3:,self.current_step]
                    else:
                        # self.simulator['xref_sim'][3:,i,k] = ref['trajectory'][3:,k+1] + gap_vel[:,i]
                        if i>0:
                            self.simulator['xref_sim'][3:,i,self.current_step] = self.ref[self.uris[i]][3:,self.current_step] + self.gap_vel[:,i]
                        else:
                            self.simulator['xref_sim'][3:,i,self.current_step] = self.ref[self.uris[i]][3:,self.current_step]

            else:                           # if there is only 1 agent => do not run game theory !!!
                self.simulator['xref_sim'][0:3,0,self.current_step] = self.ref[self.uris[0]][0:3,self.current_step]
                self.simulator['xref_sim'][3:,0,self.current_step] = self.ref[self.uris[0]][3:,self.current_step] + self.gap_vel[:,0]


            tic = time.time()
            # cf_control, landing_args = self.control_loop(self.current_step, self.ref, self.vref)
            # cf_control, landing_args = self.control_loop_lqr(self.current_step, self.ref, self.vref)
            # cf_control, landing_args = self.control_loop_lqr_game(self.current_step, self.simulator['xref_sim'], self.vref)
            cf_control, landing_args = self.control_loop_cbfqp_game(self.current_step, self.simulator['xref_sim'], self.vref)
            
            toc = time.time()
            self.comp_times.append(toc-tic)

            
            self.swarm.parallel_safe(self.apply_control, args_dict=cf_control)
            self.current_step += 1
            if self.current_step == len(self.ref[self.uris[0]][0,:]):
                self.flight_phase = "landing"
                # for i, uri in enumerate(self.uris):
                #     last_state_ref = self.ref[uri][:,0]  # make the landing point be the last point of the reference
                #     ref_land[uri] = np.array([[last_state_ref[0]], [last_state_ref[1]], [0.05], [0], [0], [0]]).reshape(1, -1)
                #     vref_land[uri] = np.array([[0], [0], [0]]).reshape(1, -1)
                # last_state_ref_ex = self.simulator['xref_sim'][:, -1, -1]  # make the landing point be the last point of the reference
                # self.ref_land[self.uris[-1]] = np.array([[last_state_ref_ex[0]], [last_state_ref_ex[1]], [0.05], [0], [0], [0]]).reshape(1, -1)
                # last_state_ref_ex = (self.state[self.drone_bodies[-1]][:, 0:6]).reshape(1, -1)  
                last_state_ref_ex = (self.simulator['xref_sim'][:, -1, -1]).ravel()  
                self.ref_land[self.uris[-1]] = np.array([[last_state_ref_ex[0]], [last_state_ref_ex[1]], [0.05], [0], [0], [0]]).reshape(1, -1)
                self.current_step = 0  # Reset step for landing phase
                self.landing_args = landing_args
                for item in self.landing_args.keys():
                    self.T_land[item] = float(self.landing_args[item][1]) / self.v_land

        elif self.flight_phase == "landing":
            cf_input = {}

            done = 1
            # Land gradually, decrease thrust to 0 iteratively
            for i, drone in enumerate(self.drone_bodies):
                currentZ = self.state[drone][:, 2][0]
                if currentZ < 0.02:
                    cf_input[self.uris[i]] = [[0, 0, 0, 0]]
                elif self.current_step * self.Ts < self.T_land[self.uris[i]]:
                    v = cfcontrol.compute_land_control(self.state[drone][:, 0:6], self.ref_land[self.uris[i]].reshape(1, -1), self.Kp, self.Kd)
                    _, _, yaw_tmp = cfcontrol.quaternion_to_euler(self.state[drone][:, 6:10][0])
                    [roll, pitch, _, _] = cfcontrol.get_cf_input(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i])
                    cf_input[self.uris[i]] = [[roll, pitch, 0, int(self.landing_args[self.uris[i]][0] * (1 - 0.1*self.current_step/self.T_land[self.uris[i]]))]]
                    done = 0
                else:
                    cf_input[self.uris[i]] = [[0, 0, 0, 0]]
                    done = 1

            if done == 0:
                self.swarm.parallel_safe(self.apply_control, args_dict=cf_input)
                self.current_step = self.current_step + 1

            if done == 1:
                # cfcontrol.save_data(f"src/Data_Drone/data_files/drone_data_{self.simid}_seq.npy", self.output_data, self.ref, self.vref)
                # cfcontrol.save_data(f"src/Data_Drone/data_files/drone_data_{self.simid}_takeoff_seq.npy", self.takeoff_data, self.refto, self.vrefto)
                # np.save(f"src/Data_Drone/data_files/computation_times_{self.simid}_seq.npy", {"comp_times": self.comp_times})

                cfcontrol.save_data(f"src/Data_Drone/drone_data_{self.simid}_seq.npy", self.output_data, self.ref, self.vref)
                # save dynamics data
                save_rds("src/Data_Drone/rds_sim_{id_file}.npy".format(id_file=self.simid), self.simulator)

                self.get_logger().info("Landing complete. Shutting down swarm.")
                self.swarm.parallel_safe(self.land)
                self.swarm.close_links()
                self.timer.cancel()

    def control_loop(self, cnt, ref_set, vref_set):
        """ Compute control inputs for each drone based on current state and references. """
        cf_control = {}
        currentTz = {}

        for i, drone in enumerate(self.drone_bodies):
            state_xi = self.state[drone][:, 0:6].T
            _, _, yaw_tmp = cfcontrol.quaternion_to_euler(self.state[drone][:, 6:10][0])

            v = compute_control_real(self.solver_data, state_xi.T, ref_set[self.uris[i]], vref_set[self.uris[i]], cnt, id=0)

            cf_control[self.uris[i]] = self.get_real_control(i, drone, v, yaw_tmp)
            currentTz[self.uris[i]] = [cf_control[self.uris[i]][0][3], self.state[drone][:, 2][0]]

            if self.flight_phase == "takeoff":
                self.takeoff_data[drone] = np.vstack((self.takeoff_data[drone],
                                                      np.hstack((np.array(self.state[drone]).reshape(1, -1),
                                                                 np.array(self.controls[drone]).reshape(1, -1),
                                                                 np.array(yaw_tmp).reshape(1, -1),
                                                                 np.array(time.time() - self.start_time).reshape(1, -1),
                                                                 np.array(v).reshape(1, -1),
                                                                 np.array(ref_set[self.uris[i]][cnt, :]).reshape(1, -1),
                                                                 np.array(vref_set[self.uris[i]][cnt, :]).reshape(1, -1)))))
            if self.flight_phase == "flight":
                self.output_data[drone] = np.vstack((self.output_data[drone],
                                                     np.hstack((np.array(self.state[drone]).reshape(1, -1),
                                                                np.array(self.controls[drone]).reshape(1, -1),
                                                                np.array(yaw_tmp).reshape(1, -1),
                                                                np.array(time.time() - self.start_time).reshape(1, -1),
                                                                np.array(v).reshape(1, -1),
                                                                np.array(ref_set[self.uris[i]][cnt, :]).reshape(1, -1),
                                                                np.array(vref_set[self.uris[i]][cnt, :]).reshape(1, -1)))))

        return cf_control, currentTz

    def control_loop_lqr(self, cnt, ref_set, vref_set):
        """ Compute control inputs for each drone based on current state and references. """
        cf_control = {}
        currentTz = {}

        for i, drone in enumerate(self.drone_bodies):
            state_xi = self.state[drone][:, 0:6].T
            _, _, yaw_tmp = cfcontrol.quaternion_to_euler(self.state[drone][:, 6:10][0])

            # v = compute_control_real(self.solver_data, state_xi.T, ref_set[self.uris[i]], vref_set[self.uris[i]], cnt, id=0)
            uri = self.uris[i]

            # shapes:
            # state_xi: (6,1)
            # ref_i:    (6,1)
            # vref_i:   (3,1)
            ref_i  = ref_set[uri][:, cnt].reshape(6, 1)   # (6,1)
            vref_i = vref_set[uri][:, cnt].reshape(3, 1)  # (3,1)
            # def compute_control(v_ref, x0, xref, Kf):
            #     v = v_ref + np.matmul(Kf, x0 - xref)
            #     return v
            # v = cfcontrol.compute_control(vrefto[uris[i]][cnt, :].reshape(-1, 1), pos_tmp, refto[uris[i]][cnt].reshape(-1,1), Kf[uris[i]])
            # cf_control[uris[i]] = [cfcontrol.get_cf_input(v, yaw_tmp, T_coeff=T_coeff[i],alpha=alpha[i], mass=m[i])]
            # controls[drone_bodies[i]] = cfcontrol.get_real_input(v,yaw_tmp)
            # v = (self.Kf[uri] @ (state_xi - ref_i) + vref_i).reshape(-1,1)
            v = (self.Kf[uri] @ (state_xi - ref_i) + vref_i + self.Ki @ self.pos_int[i]).reshape(-1,1)
            cf_control[self.uris[i]] = [cfcontrol.get_cf_input_linear(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i],PWM0=self.PWM0[i])]
            # [Roll, Pitch, Yawrate, Thrust_pwm] = cfcontrol.get_cf_input_linear(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i],PWM0=self.PWM0[i])
            # cf_control[self.uris[i]] = [[
            #     Roll + 0.055 * self.roll_int[i] + 0.125 * self.roll_p[i],
            #     Pitch + 0.055 * self.pitch_int[i] + 0.125 * self.pitch_p[i],
            #     -0.0001 * yaw_tmp * 0.0,
            #     int(Thrust_pwm + self.thrust_pwm_int[i] * 0.0055 + 0.012 * self.thrust_p[i])  # thrust in PWM unit
            # ]]
            integrate = True
            for j in range(2):
                if (v[j] >= self.cf_params[self.drone_bodies[i]].plant['amax'][j]):
                    v[j] = self.cf_params[self.drone_bodies[i]].plant['amax'][j].squeeze()
                    integrate = False
                elif (v[j] <= self.cf_params[self.drone_bodies[i]].plant['amin'][j]):
                    v[j] = self.cf_params[self.drone_bodies[i]].plant['amin'][j].squeeze()
                    integrate = False
                    
            dpos = -(state_xi[0:3] - ref_i[0:3]).reshape(-1,1)
            if integrate:
                self.pos_int[i] = dpos * self.Ts + self.pos_int[i]
            # cf_control[self.uris[i]] = self.get_real_control(i, drone, v, yaw_tmp)
            currentTz[self.uris[i]] = [cf_control[self.uris[i]][0][3], self.state[drone][:, 2][0]]

            # if self.flight_phase == "takeoff":
            #     self.takeoff_data[drone] = np.vstack((self.takeoff_data[drone],
            #                                           np.hstack((np.array(self.state[drone]).reshape(1, -1),
            #                                                      np.array(self.controls[drone]).reshape(1, -1),
            #                                                      np.array(yaw_tmp).reshape(1, -1),
            #                                                      np.array(time.time() - self.start_time).reshape(1, -1),
            #                                                      np.array(v).reshape(1, -1),
            #                                                      np.array(ref_set[self.uris[i]][cnt, :]).reshape(1, -1),
            #                                                      np.array(vref_set[self.uris[i]][cnt, :]).reshape(1, -1)))))
            # if self.flight_phase == "flight":
            #     self.output_data[drone] = np.vstack((self.output_data[drone],
            #                                          np.hstack((np.array(self.state[drone]).reshape(1, -1),
            #                                                     np.array(self.controls[drone]).reshape(1, -1),
            #                                                     np.array(yaw_tmp).reshape(1, -1),
            #                                                     np.array(time.time() - self.start_time).reshape(1, -1),
            #                                                     np.array(v).reshape(1, -1),
            #                                                     np.array(ref_set[self.uris[i]][cnt, :]).reshape(1, -1),
            #                                                     np.array(vref_set[self.uris[i]][cnt, :]).reshape(1, -1)))))

        return cf_control, currentTz

    def control_loop_lqr_game(self, cnt, ref_set, vref_set):
        """ Compute control inputs for each drone based on current state and references. """
        cf_control = {}
        currentTz = {}

        for i, drone in enumerate(self.drone_bodies):
            state_xi = self.state[drone][:, 0:6].T
            _, _, yaw_tmp = cfcontrol.quaternion_to_euler(self.state[drone][:, 6:10][0])

            # v = compute_control_real(self.solver_data, state_xi.T, ref_set[self.uris[i]], vref_set[self.uris[i]], cnt, id=0)
            uri = self.uris[i]

            # shapes:
            # state_xi: (6,1)
            # ref_i:    (6,1)
            # vref_i:   (3,1)
            # ref_i  = ref_set[uri][:, cnt].reshape(6, 1)   # (6,1)
            # vref_i = vref_set[uri][:, cnt].reshape(3, 1)  # (3,1)
            

            if (sum(self.simulator['A_sim'][-1,:,cnt])<1 or cnt<(self.Tf/4)/self.Ts) and i==self.no_drones-1:
                ref_i  = self.ref[uri][:, cnt].reshape(6, 1)   # (6,1)
                vref_i = self.vref[uri][:, cnt].reshape(3, 1)  # (3,1)
            else:
                ref_i  = ref_set[:,i, cnt].reshape(6, 1)   # (6,1)
                # vref_i = vref_set[:,i, cnt].reshape(3, 1)  # (3,1)
                vref_i = (self.vref[self.uris[0]][:, cnt] + self.gap_acc[:,i]).reshape(3, 1)  # (3,1)
                

            v = (self.Kf[uri] @ (state_xi - ref_i) + vref_i).reshape(-1,1)
            cf_control[self.uris[i]] = [cfcontrol.get_cf_input_linear(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i],PWM0=self.PWM0[i])]

            currentTz[self.uris[i]] = [cf_control[self.uris[i]][0][3], self.state[drone][:, 2][0]]



        return cf_control, currentTz

    def control_loop_cbfqp_game(self, cnt, ref_set, vref_set):
        """ Compute control inputs for each drone based on current state and references. """
        cf_control = {}
        currentTz = {}

        for i, drone in enumerate(self.drone_bodies):
            state_xi = self.state[drone][:, 0:6].T
            _, _, yaw_tmp = cfcontrol.quaternion_to_euler(self.state[drone][:, 6:10][0])

            # v = compute_control_real(self.solver_data, state_xi.T, ref_set[self.uris[i]], vref_set[self.uris[i]], cnt, id=0)
            uri = self.uris[i]

            # shapes:
            # state_xi: (6,1)
            # ref_i:    (6,1)
            # vref_i:   (3,1)
            # ref_i  = ref_set[uri][:, cnt].reshape(6, 1)   # (6,1)
            # vref_i = vref_set[uri][:, cnt].reshape(3, 1)  # (3,1)
            

            if (sum(self.simulator['A_sim'][-1,:,cnt])<1 or cnt<(self.Tf/4)/self.Ts) and i==self.no_drones-1:
                ref_i  = self.ref[uri][:, cnt].reshape(6, 1)   # (6,1)
                vref_i = self.vref[uri][:, cnt].reshape(3, 1)  # (3,1)
            else:
                ref_i  = ref_set[:,i, cnt].reshape(6, 1)   # (6,1)
                # vref_i = vref_set[:,i, cnt].reshape(3, 1)  # (3,1)
                vref_i = (self.vref[self.uris[0]][:, cnt] + self.gap_acc[:,i]).reshape(3, 1)  # (3,1)
                # if i==self.no_drones-1 and cnt<((self.Tf/2)/self.Ts):
                    # vref_i = ((ref_set[3:6, i, cnt] - ref_set[3:6, i, cnt-1])/self.Ts).reshape(3, 1)  # (3,1)
                    # vref_i = vref_i - self.vref[uri][:, cnt].reshape(3, 1)
                # else if i==0 and cnt>((self.Tf/4)/self.Ts+2):
                #     vref_i = vref_i 
                # if i==self.no_drones-1:
                #     vref_i = vref_i + ((ref_set[3:6, i, cnt] - ref_set[3:6, i, cnt-1])/self.Ts).reshape(3, 1)
            
            self.solver[self.drone_bodies[i]].update(v_ref=vref_i.reshape(-1,1),
                    x_ref=ref_i.reshape(-1,1),
                    x=state_xi.reshape(-1,1),)

            res = self.solver[self.drone_bodies[i]].prob.solve()
            


            # v = (self.Kf[uri] @ (state_xi - ref_i) + vref_i).reshape(-1,1)
            v = (self.Kf[uri] @ (state_xi - ref_i) + res.x.reshape(-1, 1) + self.Ki @ self.pos_int[i]).reshape(-1,1)

            self.simulator['u_sim'][:,i,cnt] = v.squeeze()
            self.controls[drone] = cfcontrol.get_real_input(v, yaw_tmp)
            integrate = True
            for j in range(2):
                if (self.simulator['u_sim'][j,i,cnt] >= self.cf_params[self.drone_bodies[i]].plant['amax'][j]):
                    v[j] = self.cf_params[self.drone_bodies[i]].plant['amax'][j].squeeze()
                    integrate = False
                elif (self.simulator['u_sim'][j,i,cnt] <= self.cf_params[self.drone_bodies[i]].plant['amin'][j]):
                    v[j] = self.cf_params[self.drone_bodies[i]].plant['amin'][j].squeeze()
                    integrate = False

            cf_control[self.uris[i]] = [cfcontrol.get_cf_input_linear(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i],PWM0=self.PWM0[i])]
            # [Roll, Pitch, Yawrate, Thrust_pwm] = cfcontrol.get_cf_input_linear(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i],PWM0=self.PWM0[i])
            dpos = -(state_xi[0:3] - ref_i[0:3]).reshape(-1,1)
            if integrate:
                self.pos_int[i] = dpos * self.Ts + self.pos_int[i]
            # # a small PI loop to cancel out steady state error
            # cf_control[self.uris[i]] = [[
            #     Roll + 0.055 * self.roll_int[i] + 0.125 * self.roll_p[i],
            #     Pitch + 0.055 * self.pitch_int[i] + 0.125 * self.pitch_p[i],
            #     -0.0001 * yaw_tmp * 0.0,
            #     int(Thrust_pwm + self.thrust_pwm_int[i] * 0.0055 + 0.012 * self.thrust_p[i])  # thrust in PWM unit
            # ]]

            # self.roll_int[i] = Roll * self.Ts + self.roll_int[i]
            # self.pitch_int[i] = Pitch * self.Ts + self.pitch_int[i]
            # self.thrust_pwm_int[i] = (Thrust_pwm - self.Thrust_pwm_eq[i]) * self.Ts + self.thrust_pwm_int[i]

            # self.roll_p[i] = Roll
            # self.pitch_p[i] = Pitch
            # self.thrust_p[i] = Thrust_pwm - self.Thrust_pwm_eq[i]


            currentTz[self.uris[i]] = [cf_control[self.uris[i]][0][3], self.state[drone][:, 2][0]]

            if self.flight_phase == "flight":
                self.output_data[drone] = np.vstack((self.output_data[drone],
                                                     np.hstack((np.array(self.state[drone]).reshape(1, -1),
                                                                np.array(self.controls[drone]).reshape(1, -1),
                                                                np.array(yaw_tmp).reshape(1, -1),
                                                                np.array(time.time() - self.start_time).reshape(1, -1),
                                                                np.array(v).reshape(1, -1),
                                                                np.array(ref_i).reshape(1, -1),
                                                                np.array(vref_i).reshape(1, -1)))))


        return cf_control, currentTz

    def get_real_control(self, i, drone, v, yaw_tmp):
        """ Convert desired accelerations to Crazyflie control inputs with the PI loop. """
        # [Roll, Pitch, Yawrate, Thrust_pwm] = cfcontrol.get_cf_input(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i])
        [Roll, Pitch, Yawrate, Thrust_pwm] = cfcontrol.get_cf_input_linear(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i],PWM0=self.PWM0[i])
        self.controls[drone] = cfcontrol.get_real_input(v, yaw_tmp)

        # a small PI loop to cancel out steady state error
        cf_control = [[
            Roll + 0.055 * self.roll_int[i] + 0.125 * self.roll_p[i],
            Pitch + 0.055 * self.pitch_int[i] + 0.125 * self.pitch_p[i],
            -0.0001 * yaw_tmp * 0.0,
            int(Thrust_pwm + self.thrust_pwm_int[i] * 0.0055 + 0.012 * self.thrust_p[i])  # thrust in PWM unit
        ]]

        self.roll_int[i] = Roll * self.Ts + self.roll_int[i]
        self.pitch_int[i] = Pitch * self.Ts + self.pitch_int[i]
        self.thrust_pwm_int[i] = (Thrust_pwm - self.Thrust_pwm_eq[i]) * self.Ts + self.thrust_pwm_int[i]

        self.roll_p[i] = Roll
        self.pitch_p[i] = Pitch
        self.thrust_p[i] = Thrust_pwm - self.Thrust_pwm_eq[i]

        return cf_control


def main(args=None):
    rclpy.init(args=args)
    controller = CrazyflieController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down...")
        controller.swarm.parallel_safe(controller.land)
        controller.swarm.close_links()
    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()