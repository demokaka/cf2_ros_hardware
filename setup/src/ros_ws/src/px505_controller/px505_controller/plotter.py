#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import matplotlib
matplotlib.use("TkAgg")  # or another GUI backend
import matplotlib.pyplot as plt
import numpy as np
import time

class CfPlotNode(Node):
    def __init__(self):
        super().__init__("cf_plot_node")

        # Parameters
        self.declare_parameter("odom_topic", "/crazyflie_1/odom")
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        # Data storage
        self.t0 = None
        self.time_data = []
        self.pos_data = []  # [x, y, z]
        self.vel_data = []  # [vx, vy, vz]

        # Subscriber
        self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        self.get_logger().info(f"Subscribing to odometry topic: {odom_topic}")

        # Matplotlib setup
        plt.ion()
        self.fig, (self.ax_pos, self.ax_vel) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        self.fig.suptitle("Crazyflie Position & Velocity")

        self.line_px, = self.ax_pos.plot([], [], label="x [m]")
        self.line_py, = self.ax_pos.plot([], [], label="y [m]")
        self.line_pz, = self.ax_pos.plot([], [], label="z [m]")
        self.ax_pos.set_ylabel("Position [m]")
        self.ax_pos.legend()
        self.ax_pos.grid(True)

        self.line_vx, = self.ax_vel.plot([], [], label="vx [m/s]")
        self.line_vy, = self.ax_vel.plot([], [], label="vy [m/s]")
        self.line_vz, = self.ax_vel.plot([], [], label="vz [m/s]")
        self.ax_vel.set_ylabel("Velocity [m/s]")
        self.ax_vel.set_xlabel("Time [s]")
        self.ax_vel.legend()
        self.ax_vel.grid(True)

        # Timer for plot updates: 0.1 s
        self.create_timer(0.1, self.update_plot)

    def odom_callback(self, msg: Odometry):
        # Get time
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0

        # Extract position
        p = msg.pose.pose.position
        x, y, z = p.x, p.y, p.z

        # Extract linear velocity
        v = msg.twist.twist.linear
        vx, vy, vz = v.x, v.y, v.z

        self.time_data.append(t)
        self.pos_data.append([x, y, z])
        self.vel_data.append([vx, vy, vz])

        # Optionally limit history length
        if len(self.time_data) > 1000:
            self.time_data = self.time_data[-1000:]
            self.pos_data = self.pos_data[-1000:]
            self.vel_data = self.vel_data[-1000:]

    def update_plot(self):
        if len(self.time_data) == 0:
            return

        t = np.array(self.time_data)
        pos = np.array(self.pos_data)   # shape (N, 3)
        vel = np.array(self.vel_data)   # shape (N, 3)

        # Update position lines
        self.line_px.set_data(t, pos[:, 0])
        self.line_py.set_data(t, pos[:, 1])
        self.line_pz.set_data(t, pos[:, 2])

        # Update velocity lines
        self.line_vx.set_data(t, vel[:, 0])
        self.line_vy.set_data(t, vel[:, 1])
        self.line_vz.set_data(t, vel[:, 2])

        # Rescale axes
        self.ax_pos.relim()
        self.ax_pos.autoscale_view()
        self.ax_vel.relim()
        self.ax_vel.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Small pause so matplotlib can update
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = CfPlotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
