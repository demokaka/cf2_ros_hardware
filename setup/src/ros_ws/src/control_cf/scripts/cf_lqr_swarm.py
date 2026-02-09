import os

import rclpy
import rclpy.node
from rclpy.node import Node
from rclpy import executors
from crazyflie_py import *

from ament_index_python.packages import get_package_share_directory
from crazyflie_interfaces.msg import LogDataGeneric, AttitudeSetpoint
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
import tf_transformations


import pathlib
from enum import Enum
# from copy import copy
from collections import deque
import numpy as np
import yaml

from control_cf.tracking_lqr import TrackingLQR

from functools import partial
import time

class Motors(Enum):
    MOTOR_CLASSIC = 1 # https://store.bitcraze.io/products/4-x-7-mm-dc-motor-pack-for-crazyflie-2 w/ standard props
    MOTOR_UPGRADE = 2 # https://store.bitcraze.io/collections/bundles/products/thrust-upgrade-bundle-for-crazyflie-2-x


class CrazyflieLQR(Node):
    # def __init__(self, cf_name: str, rate):
    def __init__(self, rate):
        # super().__init__(node_name='cf_lqr', namespace=cf_name)
        super().__init__(node_name='cf_lqr')

        # prefix = '/' + cf_name
        # Load parameters from YAML for the simulation
        config_file = os.path.join(get_package_share_directory('control_cf'), 'config', 'Config_Crazyflie_sim.yaml')
        self.system_parameters = self.load_params(config_file)
        self.no_drones = len(self.system_parameters['drone_bodies'])

        self.is_connected = True
        self.rate = rate
        self.motors = Motors.MOTOR_CLASSIC
        self.position = [None] * (self.no_drones)
        self.velocity = [None] * (self.no_drones)
        self.attitude = [None] * (self.no_drones)

        self.trajectory_changed = True
        self.flight_mode = 'idle'
        self.trajectory_t0 = self.get_clock().now()
        self.current_step = 0

        self.takeoff_duration = 5.0
        self.land_duration = 5.0

        self.ref, self.vref = self.load_trajectories()
        self.refto, self.vrefto = self.load_takeoff_traj()
        self.ref_land, self.vref_land = self.load_landing_point()

        self.roll_int = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.pitch_int = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.thrust_pwm_int = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.roll_p = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.pitch_p = {i: 0 for i, _ in enumerate(self.drone_bodies)}
        self.thrust_p = {i: 0 for i, _ in enumerate(self.drone_bodies)}

        # self.control_queue = None
        self.control_queue = [None]
        for i in range(1, self.no_drones + 1):
            self.control_queue.append(deque())

        self.get_logger().info('Initialization completed...')

        self.is_flying = False
        self.lqr_solver = TrackingLQR()

        self.pose_subscribers = {
            i: self.create_subscription(PoseStamped, f'/cf_{i}/pose', partial(self._pose_msg_callback, idx=i-1), 10)
            for i in range(1, self.no_drones + 1)
        }

        
        # self.create_subscription(
        #     PoseStamped,
        #     f'{prefix}/pose',
        #     self._pose_msg_callback,
        #     10)
        self.velocity_subscribers = {
            i: self.create_subscription(LogDataGeneric, f'/cf_{i}/velocity', partial(self._velocity_msg_callback, idx=i-1), 10)
            for i in range(1, self.no_drones + 1)
        }
        # self.create_subscription(
        #     LogDataGeneric,
        #     f'{prefix}/velocity',
        #     self._velocity_msg_callback,
        #     10)

        self.attitude_setpoint_publishers = {
            i: self.create_publisher(AttitudeSetpoint, f'/cf_{i}/cmd_attitude_setpoint', 10)
            for i in range(1, self.no_drones + 1)
        }
        # self.attitude_setpoint_pub = self.create_publisher(
        #     AttitudeSetpoint,
        #     f'{prefix}/cmd_attitude_setpoint',
        #     10)

        
        self.takeoffService = self.create_subscription(Empty, f'/all/lqr_takeoff', self.takeoff, 10)
        self.landService = self.create_subscription(Empty, f'/all/lqr_land', self.land, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/lqr_trajectory', self.start_trajectory, 10)
        self.hoverService = self.create_subscription(Empty, f'/all/lqr_hover', self.hover, 10)

        self.create_timer(1./self.rate, self._main_loop)
        # self.create_timer(1./10, self.lqr_loop)
        self.create_timer(self.Ts, self.lqr_loop)

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
        self.pwm0 = system_parameters['PWM0']
        self.v_land = 0.01  # m/s
        self.Tto = 10  # Takeoff time in seconds

        return system_parameters

      def load_trajectories(self):
        """ Load flight reference trajectories for each drone. """
        full_ref = trajgen.get_ref_setpoints(psi=0, Tsim=self.Tsim, dt=self.Ts, version=0)
        # full_ref = generate_traj.get_ref(0, 30, 0.1)
        ref, vref = {}, {}

        ref[self.uris[0]] = full_ref["trajectory"] + np.array([0.5, 0.0, 0.0, 0, 0, 0])
        vref[self.uris[0]] = full_ref["v_ref"]

        ref[self.uris[1]] = full_ref["trajectory"] + np.array([-0.5, -0.5, 0.0, 0, 0, 0])
        vref[self.uris[1]] = full_ref["v_ref"]

        ref[self.uris[2]] = full_ref["trajectory"] + np.array([-0.5, 0.5, 0.0, 0, 0, 0])
        vref[self.uris[2]] = full_ref["v_ref"]

        # ref[self.uris[0]] = full_ref["trajectory"]
        # vref[self.uris[0]] = full_ref["v_ref"]

        # full_ref = trajgen.get_ref_setpoints(psi=0, Tsim=self.Tsim, dt=self.Ts, version=4)
        # ref[self.uris[1]] = full_ref["trajectory"] #+ np.array([0.5, -0.5, 0.2, 0, 0, 0])
        # vref[self.uris[1]] = full_ref["v_ref"]

        # full_ref = trajgen.get_ref_setpoints(psi=0, Tsim=30, dt=self.Ts, version=3)
        # ref[self.uris[2]] = full_ref["trajectory"]
        # vref[self.uris[2]] = full_ref["v_ref"]
        return ref, vref

    def load_takeoff_traj(self):
        """ Generate reference trajectories for drone takeoff. """
        refTo, vrefTo = {}, {}
        for i, uri in enumerate(self.uris):
            full_refTo = trajgen.get_ref_setpoints_takeoff(psi=0, Tto=self.Tto, dt=self.Ts, ref=self.ref[self.uris[i]])
            refTo[uri] = full_refTo["trajectory"]
            vrefTo[uri] = full_refTo["v_ref"]
        return refTo, vrefTo

    def load_landing_point(self):
        """ Define the reference landing point for each drone. """
        ref_land, vref_land = {}, {}
        for i, uri in enumerate(self.uris):
            last_state_ref = self.ref[uri][-1]  # make the landing point be the last point of the reference
            ref_land[uri] = np.array([[last_state_ref[0]], [last_state_ref[1]], [0.05], [0], [0], [0]]).reshape(1, -1)
            vref_land[uri] = np.array([[0], [0], [0]]).reshape(1, -1)
        return ref_land, vref_land

    def _pose_msg_callback(self, msg: PoseStamped, idx: int):
        self.position[idx] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.attitude[idx] = tf_transformations.euler_from_quaternion([msg.pose.orientation.x,
                                                                msg.pose.orientation.y,
                                                                msg.pose.orientation.z,
                                                                msg.pose.orientation.w], axes='rxyz')


    def _velocity_msg_callback(self, msg: LogDataGeneric, idx: int):
        self.velocity[idx] = msg.values

    def cmd_attitude_setpoint(self, roll, pitch, yaw_rate, thrust_pwm, idx: int):
        setpoint = AttitudeSetpoint()
        setpoint.roll = roll
        setpoint.pitch = pitch
        setpoint.yaw_rate = yaw_rate
        setpoint.thrust = thrust_pwm
        # self.attitude_setpoint_pub.publish(setpoint)
        self.attitude_setpoint_publishers[idx].publish(setpoint)

    # def cmd_attitude_setpoint_swarm(self, roll, pitch, yaw_rate, thrust_pwm):
    #     for i in range(1, self.no_drones + 1):
    #         setpoint = AttitudeSetpoint()
    #         setpoint.roll = roll[i]
    #         setpoint.pitch = pitch[i]
    #         setpoint.yaw_rate = yaw_rate[i]
    #         setpoint.thrust = thrust_pwm[i]
    #         self.attitude_setpoint_publishers[i].publish(setpoint)

    def thrust_to_pwm(self, collective_thrust: float) -> int:
        # omega_per_rotor = 7460.8*np.sqrt((collective_thrust / 4.0))
        # pwm_per_rotor = 24.5307*(omega_per_rotor - 380.8359)
        collective_thrust = max(collective_thrust, 0.) #  make sure it's not negative
        if self.motors == Motors.MOTOR_CLASSIC:
            return int(max(min(24.5307*(7460.8*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))
        elif self.motors == Motors.MOTOR_UPGRADE:
            return int(max(min(24.5307*(6462.1*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))
    
    def takeoff(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'takeoff'
        for i in range(1, self.no_drones + 1):
            self.go_to_position[i] = np.array([self.position[i][0],
                                                self.position[i][1],
                                                1.0])

    def hover(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'hover'
        for i in range(1, self.no_drones + 1):
            self.go_to_position[i] = np.array([self.position[i][0],
                                                self.position[i][1],
                                                self.position[i][2]])
        # self.go_to_position = np.array([self.position[0],
        #                                 self.position[1],
        #                                 self.position[2]])

    def land(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'land'
        for i in range(1, self.no_drones + 1):
            self.go_to_position[i] = np.array([self.position[i][0],
                                                self.position[i][1],
                                                0.1])
        # self.go_to_position = np.array([self.position[0],
        #                                 self.position[1],
        #                                 0.1])        

    def start_trajectory(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'trajectory'            

    def compute_land_control(state_xi, pos_ref, Kp=np.diag([2, 2, 2]), Kd=np.diag([6, 6, 6])):
        e = pos_ref[:, 0:3] - state_xi[:, 0:3]
        e_dot = pos_ref[:, 3:6] - state_xi[:, 3:6]

        v = Kp @ e.T + Kd @ e_dot.T
        v = v.flatten()

        return [float(v[0]), float(v[1]), float(v[2])]

    def get_real_control(self, i, drone, v, yaw_tmp):
        """ Convert desired accelerations to Crazyflie control inputs with the PI loop. """
        [Roll, Pitch, Yawrate, Thrust_pwm] = cfcontrol.get_cf_input(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i])
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

    def _main_loop(self):
        if self.flight_mode == 'idle':
            return

        if not self.position or not self.velocity or not self.attitude:
            self.get_logger().warning("Empty state message.")
            return
        
        if not self.is_flying:
            self.is_flying = True
            for i in range(1, self.no_drones + 1):
                self.cmd_attitude_setpoint(0.,0.,0.,0., i)

        for i in range(1, self.no_drones + 1):
        # if self.control_queue is not None:
        #     control = self.control_queue.popleft()
        #     thrust_pwm = self.thrust_to_pwm(control[3])
        #     yawrate = -3.*(self.attitude[2])
        #     self.cmd_attitude_setpoint(control[0], 
        #                                control[1], 
        #                                yawrate, 
        #                                thrust_pwm)
            if self.control_queue[i]:
                control = self.control_queue[i].popleft()
                thrust_pwm = self.thrust_to_pwm(control[3])
                yawrate = -3.*(self.attitude[i][2])
                self.cmd_attitude_setpoint(control[0], 
                                           control[1], 
                                           yawrate, 
                                           thrust_pwm, i)

    def lqr_loop(self):
        if not self.is_flying:
            return
        
        if self.trajectory_changed:
            self.trajectory_start_position = self.position
            self.trajectory_t0 = self.get_clock().now()
            self.trajectory_changed = False
        
        t = (self.get_clock().now() - self.trajectory_t0).nanoseconds / 10.0**9


        if self.flight_mode == 'takeoff':
            cf_control, _ = self.control_loop(self.current_step, self.refto, self.vrefto)
            self.current_step += 1
            if self.current_step == len(self.refto[self.uris[0]]):
                self.flight_mode = "hover"
                self.current_step = 0  # Reset step for normal trajectory

        elif self.flight_mode == "trajectory":
            # tic = time.time()
            cf_control, landing_args = self.control_loop(self.current_step, self.ref, self.vref)
            # toc = time.time()
            # self.comp_times.append(toc-tic)

            # self.swarm.parallel_safe(self.apply_control, args_dict=cf_control)
            self.current_step += 1
            if self.current_step == len(self.ref[self.uris[0]]):
                self.flight_mode = "hover"
                self.current_step = 0  # Reset step for hovering phase
                self.landing_args = landing_args
                for item in self.landing_args.keys():
                    self.T_land[item] = float(self.landing_args[item][1]) / self.v_land
            self.control_queue[i].append(u[i])
        elif self.flight_mode == "landing":
            # cf_input = {}

            done = 1
            # Land gradually, decrease thrust to 0 iteratively
            for i, drone in enumerate(self.drone_bodies):
                # currentZ = self.state[drone][:, 2][0]
                currentZ = self.position[i][2]
                if currentZ < 0.02:
                    # cf_input[self.uris[i]] = [[0, 0, 0, 0]]
                    u[i] = [0, 0, 0, 0]
                    # done = 1
                elif self.current_step * self.Ts < self.T_land[self.uris[i]]:
                    # v = cfcontrol.compute_land_control(self.state[drone][:, 0:6], self.ref_land[self.uris[i]].reshape(1, -1), self.Kp, self.Kd)
                    v = cfcontrol.compute_land_control(self.state[drone][:, 0:6], self.ref_land[self.uris[i]].reshape(1, -1), self.Kp, self.Kd)
                    # _, _, yaw_tmp = cfcontrol.quaternion_to_euler(self.state[drone][:, 6:10][0])
                    yaw_tmp = self.attitude[i][2]
                    [roll, pitch, _, _] = cfcontrol.get_cf_input(v, yaw_tmp, T_coeff=self.T_coeff[i], alpha=self.alpha[i], mass=self.m[i])
                    # cf_input[self.uris[i]] = [[roll, pitch, 0, int(self.landing_args[self.uris[i]][0] * (1 - 0.1*self.current_step/self.T_land[self.uris[i]]))]]
                    u[i] = [roll, pitch, 0, int(self.landing_args[self.uris[i]][0] * (1 - 0.1*self.current_step/self.T_land[self.uris[i]]))]
                    done = 0
                else:
                    # cf_input[self.uris[i]] = [[0, 0, 0, 0]]
                    u[i] = [0, 0, 0, 0]
                    done = 1

            if done == 0:
                # self.swarm.parallel_safe(self.apply_control, args_dict=cf_input)
                self.current_step = self.current_step + 1
            self.control_queue[i].append(u[i])
        elif self.flight_mode == 'hover':
            for i, drone in enumerate(self.drone_bodies):
            state_xi = self.state[drone][:, 0:6].T
            # _, _, yaw_tmp = cfcontrol.quaternion_to_euler(self.state[drone][:, 6:10][0])
            yaw_tmp = self.attitude[i][2]

            v = compute_control_real(self.solver_data, state_xi.T, ref_set[self.uris[i]], vref_set[self.uris[i]], cnt, id=0)

            cf_control[self.uris[i]] = self.get_real_control(i, drone, v, yaw_tmp)

            self.control_queue[i].append(u[i])

        u = np.array([0.,0.,0.])
        self.control_queue = deque(u)


def main():
    control_update_rate = 20  # Hz                 
    n_agents = 2
    
    controller = CrazyflieLQR(control_update_rate)

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down...")
        # controller.swarm.parallel_safe(controller.land)
        # controller.swarm.close_links()
    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


    # nodes = []
    # for i in range(1, n_agents+1):
    #     nodes.append(CrazyflieLQR('cf_' + str(i), control_update_rate))
    
    # executor = executors.MultiThreadedExecutor()
    # for node in nodes:
    #     executor.add_node(node)
    #     try:
    #         while rclpy.ok():
    #             node.get_logger().info('Beginning multiagent executor, shut down with CTRL-C')
    #             executor.spin()
    #     except KeyboardInterrupt:
    #         node.get_logger().info('Keyboard interrupt, shutting down.\n')
    
    # for node in nodes:
    #     node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()