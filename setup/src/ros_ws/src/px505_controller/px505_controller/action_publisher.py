import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from px505_controller.constants import QOSP, CONTROLLER_ACTION


HELP = f"""
ActionPublisher (keyboard input)

Type one of these actions and press Enter:
  {CONTROLLER_ACTION.STAND_BY}
  {CONTROLLER_ACTION.TRACK}
  {CONTROLLER_ACTION.HOVER}
  {CONTROLLER_ACTION.TAKE_OFF}
  {CONTROLLER_ACTION.LAND}
  {CONTROLLER_ACTION.EXIT}

Shortcuts:
  s -> {CONTROLLER_ACTION.STAND_BY}
  t -> {CONTROLLER_ACTION.TAKE_OFF}
  h -> {CONTROLLER_ACTION.HOVER}
  l -> {CONTROLLER_ACTION.LAND}
  r -> {CONTROLLER_ACTION.TRACK}
  e -> {CONTROLLER_ACTION.EXIT}

Other:
  help -> print this
  quit -> stop commander node
"""


class ActionPublisher(Node):
    def __init__(self):
        super().__init__("action_publisher")
        self.pub = self.create_publisher(String, "/controller/change_action", QOSP)

        self._stop_event = threading.Event()

        self._valid_actions = {
            CONTROLLER_ACTION.STAND_BY,
            CONTROLLER_ACTION.TRACK,
            CONTROLLER_ACTION.HOVER,
            CONTROLLER_ACTION.TAKE_OFF,
            CONTROLLER_ACTION.LAND,
            CONTROLLER_ACTION.EXIT,
        }

        self._shortcuts = {
            "s": CONTROLLER_ACTION.STAND_BY,
            "r": CONTROLLER_ACTION.TRACK,
            "h": CONTROLLER_ACTION.HOVER,
            "t": CONTROLLER_ACTION.TAKE_OFF,
            "l": CONTROLLER_ACTION.LAND,
            "e": CONTROLLER_ACTION.EXIT,
        }

        self.get_logger().info("ActionPublisher ready. Publishing to /controller/change_action")
        print(HELP)

        # Run stdin reader in a background thread
        self._thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self._thread.start()

        # Timer just keeps node alive and responsive to Ctrl+C
        self.create_timer(0.1, lambda: None)

    def publish_action(self, action: str):
        msg = String()
        msg.data = action
        self.pub.publish(msg)
        self.get_logger().info(f"Published action: {action}")

    def _stdin_loop(self):
        while rclpy.ok() and not self._stop_event.is_set():
            try:
                line = sys.stdin.readline()
                if not line:  # EOF
                    break
                cmd = line.strip().lower()
            except Exception as ex:
                self.get_logger().error(f"stdin error: {ex}")
                break

            if cmd == "":
                continue

            if cmd in ("help", "?"):
                print(HELP)
                continue

            if cmd in ("quit", "q", "exit_publisher"):
                self.get_logger().info("Quitting ActionPublisher.")
                self._stop_event.set()
                # Trigger shutdown from the thread safely
                rclpy.try_shutdown()
                break

            # Map shortcut to full action
            if cmd in self._shortcuts:
                cmd = self._shortcuts[cmd]

            if cmd not in self._valid_actions:
                print(f"Unknown command: '{cmd}'. Type 'help' to see options.")
                continue

            self.publish_action(cmd)

            if cmd == CONTROLLER_ACTION.EXIT:
                self.get_logger().info("EXIT action sent. (Controller decides what to do.)")

    def destroy_node(self):
        self._stop_event.set()
        super().destroy_node()


def main():
    rclpy.init()
    node = ActionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
