from __future__ import annotations

import threading
import queue
from dataclasses import dataclass
from typing import List, Union, Optional
import numpy as np


@dataclass(frozen=True)
class WaypointCommand:
    agent_name: str
    position: np.ndarray
    is_initial: bool = False
    dt: Optional[float] = None  # relative time (seconds) from previous knot to this waypoint


@dataclass(frozen=True)
class GenerateCommand:
    agent_name: str


@dataclass(frozen=True)
class PlotCommand:
    agent_name: str


@dataclass(frozen=True)
class ClearCommand:
    agent_name: str


@dataclass(frozen=True)
class LoadCommand:
    path: str


@dataclass(frozen=True)
class SendCommand:
    agent_name: str


@dataclass(frozen=True)
class QuitCommand:
    pass

@dataclass(frozen=True)
class ListTrajectoriesCommand:
    pass


KeyboardEvent = Union[
    WaypointCommand,
    GenerateCommand,
    PlotCommand,
    ClearCommand,
    LoadCommand,
    SendCommand,
    QuitCommand,
    ListTrajectoriesCommand,
]


class KeyboardInput:
    """Reads commands from stdin in a background thread."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._q: "queue.Queue[KeyboardEvent]" = queue.Queue()
        self._known_agents: List[str] = []

    def set_known_agents(self, agents: List[str]) -> None:
        self._known_agents = list(agents)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll(self) -> List[KeyboardEvent]:
        out: List[KeyboardEvent] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        self._print_help()
        while not self._stop.is_set():
            try:
                print("> ", end="", flush=True)
                line = input().strip()
            except (EOFError, KeyboardInterrupt):
                self._q.put(QuitCommand())
                return

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("help", "?"):
                self._print_help()
                continue

            if cmd == "quit":
                self._q.put(QuitCommand())
                return

            if cmd == "agents":
                print("Known agents:", self._known_agents)
                continue

            if cmd == "init":
                if len(parts) != 5:
                    print("Usage: init <agent> x y z")
                    continue
                agent = parts[1]
                if self._known_agents and agent not in self._known_agents:
                    print(f"Unknown agent '{agent}'. Type 'agents'.")
                    continue
                pos = self._parse_xyz(parts[2:5])
                if pos is None:
                    continue
                self._q.put(WaypointCommand(agent_name=agent, position=pos, is_initial=True, dt=None))
                continue

            if cmd == "wp":
                # Timed-only mode:
                #   wp <agent> x y z dt
                if len(parts) != 6:
                    print("Usage: wp <agent> x y z dt")
                    continue
                
                agent = parts[1]
                if self._known_agents and agent not in self._known_agents:
                    print(f"Unknown agent '{agent}'. Type 'agents'.")
                    continue
                
                pos = self._parse_xyz(parts[2:5])
                if pos is None:
                    continue
                try:
                    dt = float(parts[5])
                    if dt <= 0.0:
                        print("dt must be > 0 (seconds). Example: wp crazyflie_1 1 0 1 2.5")
                        continue
                except ValueError:
                    print("Invalid dt. Example: wp crazyflie_1 1 0 1 2.5")
                    continue

                self._q.put(WaypointCommand(agent_name=agent, position=pos, is_initial=False, dt=dt))
                continue

            if cmd == "load":
                # load <name>|<name>.traj|<relative>|<absolute>
                if len(parts) != 2:
                    print("Usage: load <name>|<path>")
                    continue
                self._q.put(LoadCommand(path=parts[1]))
                continue

            if cmd == "gen":
                if len(parts) != 2:
                    print("Usage: gen <agent>|all")
                    continue
                target = parts[1]
                if target != "all":
                    if self._known_agents and target not in self._known_agents:
                        print(f"Unknown agent '{target}'. Type 'agents'.")
                        continue
                self._q.put(GenerateCommand(agent_name=target))
                continue

            if cmd == "plot":
                if len(parts) != 2:
                    print("Usage: plot <agent>")
                    continue
                agent = parts[1]
                if self._known_agents and agent not in self._known_agents:
                    print(f"Unknown agent '{agent}'. Type 'agents'.")
                    continue
                self._q.put(PlotCommand(agent_name=agent))
                continue

            if cmd == "clear":
                if len(parts) != 2:
                    print("Usage: clear <agent>")
                    continue
                agent = parts[1]
                if self._known_agents and agent not in self._known_agents:
                    print(f"Unknown agent '{agent}'. Type 'agents'.")
                    continue
                self._q.put(ClearCommand(agent_name=agent))
                continue

            if cmd in ("trajs", "ls"):
                self._q.put(ListTrajectoriesCommand())
                continue

            if cmd == "send":
                if len(parts) != 2:
                    print("Usage: send <agent>|all")
                    continue
                target = parts[1]
                if target != "all":
                    if self._known_agents and target not in self._known_agents:
                        print(f"Unknown agent '{target}'. Type 'agents'.")
                        continue
                self._q.put(SendCommand(agent_name=target))
                continue

            print("Unknown command. Type 'help'.")

    def _parse_xyz(self, xyz_parts) -> Optional[np.ndarray]:
        try:
            x, y, z = map(float, xyz_parts)
            return np.array([x, y, z], dtype=float)
        except ValueError:
            print("Invalid numbers. Example: wp crazyflie_1 1 0 1 2.5")
            return None

    def _print_help(self) -> None:
        print(
            "\nCommands:\n"
            "  agents\n"
            "  init <agent> x y z\n"
            "  wp   <agent> x y z dt\n"
            "       (dt is relative segment time in seconds)\n"
            "  load <name>|<path>\n"
            "       (looks in share/control_framework/trajectory/trajectories/*.traj when given a name)\n"
            "  trajs\n"
            "  gen  <agent>|all\n"
            "  plot <agent>\n"
            "  clear <agent>\n"
            "  send <agent>|all\n"
            "  help\n"
            "  quit\n"
        )