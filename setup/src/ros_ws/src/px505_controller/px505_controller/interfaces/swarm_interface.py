from cflib.crazyflie import Crazyflie
from cflib.crazyflie.swarm import Swarm, CachedCfFactory
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import cflib.crtp
from rclpy.node import get_logger
import numpy as np
from typing import List
from time import sleep
from px505_controller.constants import g

class SwarmInterface:
    '''
    Interface to manage a swarm of Crazyflie drones.
    '''
    def __init__(self, uris: List[str], logger_name: str):
        self.uris = uris
        self.logger_name = logger_name
        
    def start(self):
        try:
            cflib.crtp.init_drivers(enable_debug_driver=False)
            factory = CachedCfFactory(rw_cache='./cache')

            self.swarm = Swarm(self.uris, factory=factory)
            self.swarm.open_links()
            self.swarm.reset_estimators()
            get_logger(self.logger_name).info("Crazyflie swarm links opened and estimators reset.")

            self.swarm.parallel_safe(self.__wait_for_param_download)
            self.swarm.sequential(self.__unlock_safety_check)

            get_logger(self.logger_name).info("Crazyflie swarm initialized successfully.")
        except Exception as e:
            get_logger(self.logger_name).error(f"Failed to start Crazyflie swarm: {e}")
            # self.shutdown_swarm()
        
    def stop(self):
        self.swarm.close_links()
        get_logger(self.logger_name).info("Crazyflie swarm links closed.")

    def send_control_inputs(self, control_inputs: dict):
        '''
        Send control inputs to all Crazyflies in the swarm.
        control_inputs: dict mapping URI to inputs dict
        control_inputs: {
            URI0 : inputs,
            URI1 : inputs
            ...
        }
        inputs: {
            'ax'  : float,
            'ay'  : float,
            'az'  : float,
            'yaw' : float,
            'mass': float
        }
        '''
        def send_input(scf: SyncCrazyflie, inputs: dict):                
            [thrust, roll, pitch] = self.get_cf_input(
                np.array([inputs['ax'], inputs['ay'], inputs['az']]),
                inputs['yaw'],
                inputs['mass']
            )

            # print(f"Converted thrust: {thrust}")

            scf.cf.commander.send_setpoint(
                roll,       # roll
                pitch,      # pitch
                0,          # yaw rate
                thrust      # thrust
            )
        
        self.swarm.parallel_safe(send_input, control_inputs)
        
    def __unlock_safety_check(self, scf: SyncCrazyflie):
        get_logger(self.logger_name).info(f"Initiated safety check for {scf.cf.link_uri}.")
        scf.cf.commander.send_setpoint(0, 0, 0, 0)
        get_logger(self.logger_name).info(f"Safety check unlocked for {scf.cf.link_uri}.")
        
    def __wait_for_param_download(self, scf: SyncCrazyflie):
        get_logger(self.logger_name).info(f"Initiated parameter download for {scf.cf.link_uri}.")
        while not scf.cf.param.is_updated:
            sleep(0.1)
            get_logger(self.logger_name).info(f"Waiting for parameters to be downloaded for {scf.cf.link_uri}...")
            pass
        get_logger(self.logger_name).info(f"Parameters downloaded for {scf.cf.link_uri}.")
        
    def __feedback_linearization_input(self, v: np.ndarray, yaw: float) -> list:
        """
        Feedback linearization laws to compute the real input from the virtual one (thrust and desired angles).
        :param v: The virtual input (accelerations).
        :param yaw: Measured yaw angle (scalar, radians)
        :return: Thrust (T), Desired roll (phi_d) and pitch (theta_d) angles
        """
        T = np.round(np.sqrt(v[0]**2 + v[1]**2 + (v[2] + g)**2), 5)
        phi_d = np.round(np.arcsin((v[0] * np.sin(yaw) - v[1] * np.cos(yaw)) / T), 5)
        theta_d = np.round(np.arctan2(v[0] * np.cos(yaw) + v[1] * np.sin(yaw), v[2] + g), 5)

        # turn radians into degrees
        phi_d = np.degrees(phi_d)
        theta_d = np.degrees(theta_d)

        return T, phi_d, theta_d          
    
    def Thrust_to_PWM_quad(self, Thrust,m=33.0,PWM0=42050):
        a1 = 2.130295e-11
        a2 = 1.032633e-6
        a3 = 5.484560e-4
        K0 = a1*PWM0**2 + a2*PWM0 + a3
        pwm_signal = (-a2 + np.sqrt(a2**2 + 4*a1*(Thrust * K0 - a3)))/(2*a1) 
        return pwm_signal
    
    def get_cf_input(self, v, yaw, mass=30.0) :
        """
        Compute the control input for the Crazyflie based on the desired virtual input (accelerations) and current yaw.
        This method applies feedback linearization to compute the required thrust and desired angles, then converts the thrust to a PWM command using the hover thrust conversion.
        :param v: The virtual input (accelerations).
        :param yaw: Measured yaw angle (scalar, radians)
        :param mass: Mass of the Crazyflie in grams (default 30g)
        :return: List containing desired thrust (PWM value), roll (phi_d), pitch (theta_d)

        """
        [T, phi_d, theta_d] = self.__feedback_linearization_input(v, yaw)
        get_logger(self.logger_name).info(f"Computed feedback linearization output - Thrust: {T}, Roll: {phi_d}, Pitch: {theta_d}")

        thrust_g = (T / g) * mass  # Convert to grams
        T = self.thrust_to_pwm(thrust_g)
        get_logger(self.logger_name).info(f"Converted thrust to PWM - Thrust (g): {thrust_g}, Thrust (PWM): {T}")

        # Return formatted for Crazyflie [Roll, Pitch, Yawrate, Thrust]
        return [int(np.clip(T, 0, 65535)), phi_d, theta_d]
    
    def thrust_to_pwm(self, thrust_g: float) -> int:
        """
        Converts thrust (grams) to 16-bit PWM using a 2nd degree polynomial fit.
        Coefficients: [-7.36578582, 1465.72737286, 1098.21200142]
        Uses Crazyflie motor test data for 16-bit PWM range (0-65535).
        """
        # y = Ax^2 + Bx + C
        a = -7.36578582
        b = 1465.72737286
        c = 1098.21200142
        
        pwm = (a * (thrust_g**2)) + (b * thrust_g) + c
        
        # Constrain the result to 16-bit range
        return int(max(0, min(65535, pwm)))