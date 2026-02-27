from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from ament_index_python.packages import get_package_share_directory

from control_framework.config import load_config
from control_framework.config.models.global_config import GlobalConfig
from control_framework.config.models.agents import EnvAgentsConfig, CrazyflieConfig
from control_framework.config.models.controllers import EnvControllersConfig

from control_framework.trajectory.input import (
    KeyboardInput,
    KeyboardEvent,
    WaypointCommand,
    GenerateCommand,
    PlotCommand,
    ClearCommand,
    LoadCommand,
    SendCommand,
    ListTrajectoriesCommand,
    QuitCommand,
)
# from control_framework.trajectory.plotter import TrajectoryPlotter
from control_framework.trajectory.generator import TrajectoryGenerator

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

@dataclass(frozen=True)
class TimedWaypoint:
    position: np.ndarray
    dt: float  # relative segment time (seconds) from previous knot


@dataclass
class AgentStorage:
    name: str
    cfg: CrazyflieConfig
    dt: float
    x0: np.ndarray = field(default_factory=lambda: np.zeros((9,), dtype=float))
    waypoints: List[TimedWaypoint] = field(default_factory=list)
    traj: np.ndarray = field(default_factory=lambda: np.empty((0, 9), dtype=float))
    gen: Optional[TrajectoryGenerator] = None


class TrajectoryNode(Node):
    """Offline trajectory generation node driven by keyboard."""

    def __init__(self):
        super().__init__("trajectory_node")

        self.declare_parameter("env", "hitl")
        self.env: str = self.get_parameter("env").value

        self.config: GlobalConfig = load_config()
        self.agents_config: EnvAgentsConfig
        self.controllers_config: EnvControllersConfig

        self.agents: Dict[str, AgentStorage] = {}

        self.keyboard = KeyboardInput()
        # self.plotter = TrajectoryPlotter()

        # Outbox for "send" command
        self._outbox: Dict[str, np.ndarray] = {}

        # Publishers per agent: /<agent>/trajectory
        self._traj_pub: Dict[str, rclpy.publisher.Publisher] = {}

        self._load()

        # Create publishers now that we know agent names
        for name in self.agents.keys():
            self._traj_pub[name] = self.create_publisher(
                Float64MultiArray,
                f"/{name}/trajectory",
                QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT
            )
        )

        self.keyboard.set_known_agents(sorted(self.agents.keys()))
        self.keyboard.start()

        self.create_timer(0.05, self._tick)

    def _load(self) -> None:
        if self.env == "sitl":
            self.agents_config = self.config.agents.sitl
            self.controllers_config = self.config.controllers.sitl
        else:
            self.agents_config = self.config.agents.hitl
            self.controllers_config = self.config.controllers.hitl

        id_to_name = self.agents_config.id_to_name

        dt_by_agent: Dict[str, float] = {}
        for _, ctrl_cfg in self.controllers_config.items.items():
            agent_name = id_to_name[int(ctrl_cfg.controlled_agent_id)]
            dt_by_agent[agent_name] = float(ctrl_cfg.parameters.dt) / 1000.0

        for agent_name, cfg in self.agents_config.agents.items():
            dt = dt_by_agent.get(agent_name, 0.02)
            self.agents[agent_name] = AgentStorage(name=agent_name, cfg=cfg, dt=dt)

        self._log(f"[{self.env}] agents={list(self.agents.keys())}")

    def _tick(self) -> None:
        for ev in self.keyboard.poll():
            if isinstance(ev, QuitCommand):
                self.destroy_node()
                return
            if isinstance(ev, LoadCommand):
                self._on_load(ev)
            elif isinstance(ev, WaypointCommand):
                self._on_waypoint(ev)
            elif isinstance(ev, ClearCommand):
                self._on_clear(ev)
            elif isinstance(ev, GenerateCommand):
                self._on_generate(ev)
            elif isinstance(ev, PlotCommand):
                self._on_plot(ev)
            elif isinstance(ev, ListTrajectoriesCommand):
                self._on_list_trajectories()
            elif isinstance(ev, SendCommand):
                self._on_send(ev)

    def _on_waypoint(self, ev: WaypointCommand) -> None:
        a = self.agents[ev.agent_name]
        if ev.is_initial:
            a.x0 = np.zeros((9,), dtype=float)
            a.x0[0:3] = ev.position
        else:
            if ev.dt is None or float(ev.dt) <= 0.0:
                self._log(f"[{a.name}] wp requires dt > 0 (seconds)")
                return
            a.waypoints.append(TimedWaypoint(position=ev.position, dt=float(ev.dt)))

    def _on_list_trajectories(self) -> None:
        d = self._trajectory_dir()
        if not d.exists():
            self._log(f"trajectory folder not found: {d}")
            return

        files = sorted([p.name for p in d.glob("*.traj")])
        if not files:
            self._log(f"no .traj files found in {d}")
            return

        self._log(f"trajectories in {d}: {files}")

    def _trajectory_dir(self) -> Path:
        share = Path(get_package_share_directory("control_framework"))
        return share / "trajectory" / "trajectories"

    def _resolve_traj_path(self, spec: str) -> str:
        """
        Resolve trajectory file path.

        - If spec is absolute and exists -> use it.
        - Else search in <share>/trajectory/trajectories/
          - allow "takeoff" or "takeoff.traj"
        """
        p = Path(spec)

        if p.is_absolute():
            return str(p)

        name = spec if spec.endswith(".traj") else (spec + ".traj")
        candidate = self._trajectory_dir() / name
        return str(candidate)

    def _on_load(self, ev: LoadCommand) -> None:
        path = self._resolve_traj_path(ev.path)
        self._apply_command_file(path)

    def _on_clear(self, ev: ClearCommand) -> None:
        a = self.agents[ev.agent_name]
        a.waypoints.clear()
        a.traj = np.empty((0, 9), dtype=float)
        if a.gen is not None:
            a.gen.clear_waypoints()
        try:
            self._outbox.pop(a.name, None)
        except Exception:
            pass

    def _on_generate(self, ev: GenerateCommand) -> None:
        if ev.agent_name == "all":
            for a in self.agents.values():
                if not a.waypoints:
                    continue

                gen = self._generator(a)
                gen.clear_waypoints()

                gen.add_waypoint(a.x0[0:3])

                for twp in a.waypoints:
                    gen.add_waypoint(twp.position, dt=twp.dt)

                a.traj = gen.generate(x0=a.x0)
                self._log(f"[{a.name}] generated traj {a.traj.shape}")
            return

        a = self.agents[ev.agent_name]
        gen = self._generator(a)

        gen.clear_waypoints()

        gen.add_waypoint(a.x0[0:3])

        for twp in a.waypoints:
            gen.add_waypoint(twp.position, dt=twp.dt)

        a.traj = gen.generate(x0=a.x0)
        self._log(f"[{a.name}] generated traj {a.traj.shape}")

    def _on_plot(self, ev: PlotCommand) -> None:
        # a = self.agents[ev.agent_name]
        # wps = np.array([w.position for w in a.waypoints], dtype=float) if a.waypoints else None
        # self.plotter.plot(a.name, a.traj, a.dt, wps)
        return

    def _traj_to_msg(self, traj: np.ndarray, dt: float) -> Float64MultiArray:
        """
        Pack traj (N+1, 9) into Float64MultiArray.

        layout:
          dim[0] = points (N+1)
          dim[1] = features (9)
        data:
          row-major flatten: [traj[0,0..8], traj[1,0..8], ...]
        """
        if traj.ndim != 2 or traj.shape[1] != 9:
            raise ValueError(f"traj must be (N,9) got {traj.shape}")

        msg = Float64MultiArray()

        d0 = MultiArrayDimension()
        d0.label = "points"
        d0.size = int(traj.shape[0])
        d0.stride = int(traj.shape[0] * traj.shape[1])

        d1 = MultiArrayDimension()
        d1.label = "features"
        d1.size = int(traj.shape[1])
        d1.stride = int(traj.shape[1])

        msg.layout.dim = [d0, d1]
        msg.layout.data_offset = 0

        # Optional: include dt as an extra element at the beginning?
        # Keeping it simple: don't embed dt; dt is implied by the controller config.
        msg.data = traj.astype(np.float64, copy=False).reshape(-1).tolist()
        return msg

    def _publish_traj(self, agent_name: str, traj: np.ndarray, dt: float) -> None:
        pub = self._traj_pub.get(agent_name)
        if pub is None:
            self._log(f"[{agent_name}] no publisher for /{agent_name}/trajectory")
            return

        msg = self._traj_to_msg(traj, dt)
        pub.publish(msg)

        duration = max((traj.shape[0] - 1) * float(dt), 0.0)
        self._log(
            f"[{agent_name}] published /{agent_name}/trajectory: points={traj.shape[0]} dt={dt:.4f}s duration={duration:.3f}s"
        )

    def _on_send(self, ev: SendCommand) -> None:
        """
        Supports command to send the trajectory for one or multiple drones.

        Uses Float64MultiArray publishing on:
          /<agent>/trajectory
        """
        def _send_one(a: AgentStorage) -> None:
            if a.traj is None or a.traj.size == 0:
                self._log(f"[{a.name}] no trajectory to send (generate first)")
                return

            self._outbox[a.name] = a.traj.copy()
            self._publish_traj(a.name, a.traj, a.dt)

        if ev.agent_name == "all":
            a = self.agents[ev.agent_name]
            for a in self.agents.values():
                _send_one(a)
            return

        a = self.agents[ev.agent_name]
        _send_one(a)

    def _apply_command_file(self, path: str) -> None:
        deferred = []  # list[tuple[str, list[str]]] of (cmd, parts)

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                cmd = parts[0].lower()

                # Phase 1: apply state edits now
                if cmd == "init":
                    if len(parts) != 5:
                        continue
                    agent_name = parts[1]
                    if agent_name not in self.agents:
                        continue

                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    p = np.array([x, y, z], dtype=float)

                    a = self.agents[agent_name]
                    a.x0 = np.zeros((9,), dtype=float)
                    a.x0[0:3] = p
                    continue

                if cmd == "wp":
                    if len(parts) != 6:   # timed-only
                        continue
                    agent_name = parts[1]
                    if agent_name not in self.agents:
                        continue

                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    dt = float(parts[5])
                    if dt <= 0.0:
                        continue

                    p = np.array([x, y, z], dtype=float)
                    a = self.agents[agent_name]
                    a.waypoints.append(TimedWaypoint(position=p, dt=dt))
                    continue

                if cmd == "clear":
                    if len(parts) != 2:
                        continue
                    agent_name = parts[1]
                    if agent_name not in self.agents:
                        continue
                    self._on_clear(ClearCommand(agent_name=agent_name))
                    continue

                # Phase 2: defer actions until after summary log
                if cmd in ("gen", "plot", "send", "quit", "load"):
                    deferred.append((cmd, parts))
                    continue

                # ignore unknowns
                continue

        # Log summary BEFORE running gen/plot/send
        summary = {name: len(a.waypoints) for name, a in self.agents.items()}
        self._log(f"loaded file: {path} wps={summary}")

        # Now run deferred actions in file order
        for cmd, parts in deferred:
            if cmd == "gen":
                if len(parts) != 2:
                    continue
                target = parts[1]
                if target != "all" and target not in self.agents:
                    continue
                self._on_generate(GenerateCommand(agent_name=target))

            elif cmd == "plot":
                if len(parts) != 2:
                    continue
                agent_name = parts[1]
                if agent_name not in self.agents:
                    continue
                self._on_plot(PlotCommand(agent_name=agent_name))

            elif cmd == "send":
                if len(parts) != 2:
                    continue
                target = parts[1]
                if target != "all" and target not in self.agents:
                    continue
                self._on_send(SendCommand(agent_name=target))

            elif cmd == "load":
                # allow nested loads (optional) from share folder by name
                if len(parts) != 2:
                    continue
                self._on_load(LoadCommand(path=parts[1]))

            elif cmd == "quit":
                self.destroy_node()
                return

    def _generator(self, a: AgentStorage) -> TrajectoryGenerator:
        if a.gen is not None:
            return a.gen

        R = np.diag([1.0, 1.0, 1.0])
        x_min = np.array([-2.0, -2.0, -0.1, 
                          -5.0, -5.0, -5.0, 
                          -2.0, -2.0, -2.0], dtype=float)
        x_max = np.array([+2.0, +2.0, +2.0, 
                          +5.0, +5.0, +5.0,
                          +2.0, +2.0, +2.0], dtype=float)

        a.gen = TrajectoryGenerator(
            dt=a.dt,
            R=R,
            x_min=x_min,
            x_max=x_max,
        )
        return a.gen

    def destroy_node(self):
        try:
            self.keyboard.stop()
        except Exception:
            pass
        try:
            self.plotter.stop()
        except Exception:
            pass
        super().destroy_node()

    def _log(self, message: str):
        self.get_logger().info(message)
        print("> ", end="", flush=True)


def main(args=None):
    rclpy.init(args=args)

    node = TrajectoryNode()

    try:
        rclpy.spin(node, executor=MultiThreadedExecutor(num_threads=10))
    except KeyboardInterrupt:
        ### nothing to treat
        pass
    finally:
        ### keyboard interrupt destroys everything automatically and all contexts are invalidated...
        pass


if __name__ == "__main__":
    main()