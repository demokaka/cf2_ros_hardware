#!/usr/bin/env python3
import threading
import time
import os
import csv
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Accel
from nav_msgs.msg import Odometry

class RealtimePlotNode(Node):
    def __init__(self, history_sec: float = 10.0):
        super().__init__("cf_realtime_plot_two_figs")
        self.history_sec = float(history_sec)
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()

        # --- LOGGING SETUP ---
        self.log_dir = "flight_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.control_log_path = os.path.join(self.log_dir, f"control_{stamp}.csv")
        self.odom_log_path = os.path.join(self.log_dir, f"odom_{stamp}.csv")

        # Initialize CSV headers
        self._init_csv(self.control_log_path, ["t", "thrust_pwm", "roll_deg", "pitch_deg"])
        self._init_csv(self.odom_log_path, ["t", "p_x", "p_y", "p_z", "v_x", "v_y", "v_z"])

        self.get_logger().info(f"Logging Control to: {self.control_log_path}")
        self.get_logger().info(f"Logging Odometry to: {self.odom_log_path}")

        # Data storage for plotting (Deques)
        self.maxlen = 20000
        self.t = deque(maxlen=self.maxlen)
        self.thrust_pwm = deque(maxlen=self.maxlen)
        self.roll_deg = deque(maxlen=self.maxlen)
        self.pitch_deg = deque(maxlen=self.maxlen)
        
        self.p_x = deque(maxlen=self.maxlen); self.p_y = deque(maxlen=self.maxlen); self.p_z = deque(maxlen=self.maxlen)
        self.v_x = deque(maxlen=self.maxlen); self.v_y = deque(maxlen=self.maxlen); self.v_z = deque(maxlen=self.maxlen)

        # Subs
        self.create_subscription(Accel, "/crazyflie_2/control_input", self._cb_u, 10)
        self.create_subscription(Odometry, "/crazyflie_2/odom", self._cb_odom, 10)

    def _init_csv(self, path, headers):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(headers)

    def _now(self) -> float:
        return time.perf_counter() - self.t0

    def _trim_history(self):
        # Keep only last history_sec for the live plot windows
        while self.t and (self.t[-1] - self.t[0] > self.history_sec):
            self.t.popleft()
            self.thrust_pwm.popleft(); self.roll_deg.popleft(); self.pitch_deg.popleft()
            self.p_x.popleft(); self.p_y.popleft(); self.p_z.popleft()
            self.v_x.popleft(); self.v_y.popleft(); self.v_z.popleft()

    def _cb_u(self, msg: Accel):
        t_now = self._now()
        u = [float(msg.linear.x), float(msg.linear.y), float(msg.linear.z)]
        
        with self.lock:
            # 1. Log to File (Control)
            with open(self.control_log_path, 'a', newline='') as f:
                csv.writer(f).writerow([t_now] + u)

            # 2. Update deques for plotting
            self.t.append(t_now)
            self.thrust_pwm.append(u[0])
            self.roll_deg.append(u[1])
            self.pitch_deg.append(u[2])
            
            # Fill odom deques with last known value to keep lengths equal for plotting
            last_p = [self.p_x[-1], self.p_y[-1], self.p_z[-1]] if self.p_x else [0.0]*3
            last_v = [self.v_x[-1], self.v_y[-1], self.v_z[-1]] if self.v_x else [0.0]*3
            self.p_x.append(last_p[0]); self.p_y.append(last_p[1]); self.p_z.append(last_p[2])
            self.v_x.append(last_v[0]); self.v_y.append(last_v[1]); self.v_z.append(last_v[2])
            
            self._trim_history()

    def _cb_odom(self, msg: Odometry):
        t_now = self._now()
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        p_list = [float(p.x), float(p.y), float(p.z)]
        v_list = [float(v.x), float(v.y), float(v.z)]

        with self.lock:
            # 1. Log to File (Odometry)
            with open(self.odom_log_path, 'a', newline='') as f:
                csv.writer(f).writerow([t_now] + p_list + v_list)

            # 2. Update deques for plotting
            self.t.append(t_now)
            self.p_x.append(p_list[0]); self.p_y.append(p_list[1]); self.p_z.append(p_list[2])
            self.v_x.append(v_list[0]); self.v_y.append(v_list[1]); self.v_z.append(v_list[2])
            
            # Fill control deques with last known value for plot alignment
            last_u = [self.thrust_pwm[-1], self.roll_deg[-1], self.pitch_deg[-1]] if self.thrust_pwm else [0.0]*3
            self.thrust_pwm.append(last_u[0]); self.roll_deg.append(last_u[1]); self.pitch_deg.append(last_u[2])
            
            self._trim_history()

def main():
    rclpy.init()
    node = RealtimePlotNode(history_sec=10.0)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    plt.ion()

    # --- Figure 1: Control ---
    fig_u, (ax_thr, ax_ang) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig_u.canvas.manager.set_window_title("Control Inputs")
    (l_thr,) = ax_thr.plot([], [], label="Thrust PWM")
    (l_roll,) = ax_ang.plot([], [], label="Roll")
    (l_pitch,) = ax_ang.plot([], [], label="Pitch")
    for ax in [ax_thr, ax_ang]: ax.grid(True); ax.legend(loc="upper right")

    # --- Figure 2: Odom ---
    fig_o, (ax_pos, ax_vel) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig_o.canvas.manager.set_window_title("Odometry")
    (lp_x,) = ax_pos.plot([], [], label="x"); (lp_y,) = ax_pos.plot([], [], label="y"); (lp_z,) = ax_pos.plot([], [], label="z")
    (lv_x,) = ax_vel.plot([], [], label="vx"); (lv_y,) = ax_vel.plot([], [], label="vy"); (lv_z,) = ax_vel.plot([], [], label="vz")
    for ax in [ax_pos, ax_vel]: ax.grid(True); ax.legend(loc="upper right")

    try:
        while plt.fignum_exists(fig_u.number) and plt.fignum_exists(fig_o.number):
            with node.lock:
                t = list(node.t)
                thr, roll, pitch = list(node.thrust_pwm), list(node.roll_deg), list(node.pitch_deg)
                px, py, pz = list(node.p_x), list(node.p_y), list(node.p_z)
                vx, vy, vz = list(node.v_x), list(node.v_y), list(node.v_z)

            if len(t) >= 2:
                l_thr.set_data(t, thr); l_roll.set_data(t, roll); l_pitch.set_data(t, pitch)
                lp_x.set_data(t, px); lp_y.set_data(t, py); lp_z.set_data(t, pz)
                lv_x.set_data(t, vx); lv_y.set_data(t, vy); lv_z.set_data(t, vz)
                
                for ax in [ax_thr, ax_ang, ax_pos, ax_vel]:
                    ax.relim(); ax.autoscale_view(); ax.set_xlim(t[0], t[-1])

            plt.pause(0.03)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()