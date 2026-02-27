from __future__ import annotations
import threading
import queue
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class TrajectoryPlotter:
    """Shows plots in a dedicated thread."""

    def __init__(self):
        self._q: "queue.Queue[tuple[str, np.ndarray, float, np.ndarray | None]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._q.put(("", np.empty((0, 0)), 0.0, None))

    def plot(self, agent_name: str, traj: np.ndarray, dt: float, waypoints: np.ndarray | None = None) -> None:
        if traj is None or traj.size == 0:
            return
        self._q.put((agent_name, traj, dt, waypoints))

    def _run(self) -> None:

        while not self._stop.is_set():
            agent_name, traj, dt, waypoints = self._q.get()
            if self._stop.is_set():
                break
            if traj.size == 0:
                continue

            t = np.arange(traj.shape[0]) * dt
            pos = traj[:, 0:3]
            vel = traj[:, 3:6]
            acc = traj[:, 6:9]

            plt.figure()
            plt.plot(t, pos[:, 0], label="x")
            plt.plot(t, pos[:, 1], label="y")
            plt.plot(t, pos[:, 2], label="z")
            plt.grid(True)
            plt.legend()
            plt.title(f"{agent_name} pos")

            plt.figure()
            plt.plot(t, vel[:, 0], label="vx")
            plt.plot(t, vel[:, 1], label="vy")
            plt.plot(t, vel[:, 2], label="vz")
            plt.grid(True)
            plt.legend()
            plt.title(f"{agent_name} vel")

            plt.figure()
            plt.plot(t, acc[:, 0], label="ax")
            plt.plot(t, acc[:, 1], label="ay")
            plt.plot(t, acc[:, 2], label="az")
            plt.grid(True)
            plt.legend()
            plt.title(f"{agent_name} acc")

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="traj")
            if waypoints is not None and waypoints.size:
                ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], marker="x", s=80, label="wps")
            ax.legend()
            ax.set_title(f"{agent_name} 3d")

            plt.show()