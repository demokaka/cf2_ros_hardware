import os
from ament_index_python import get_package_share_directory
from px505_controller.controller import ControllerNode
from px505_controller.control_laws.pid import PIDControlLaw, PIDControlLawSetting
from px505_controller.control_laws.models import DoubleIntegrator3DOF
from px505_controller.control_laws.lqr import LQRControlLaw, LQRControlLawSetting
from px505_controller.control_laws.mpc import MPCControlLaw, MPCControlLawSetting
from px505_controller.config.load import load_config, GlobalConfig
from px505_controller.config.models.controllers import Controller
from px505_controller.control_laws.control_law import ControlLaw, ControlLawSetting
from px505_controller.interfaces.crazyflie_descriptor import CrazyflieDescriptor
import rclpy
import numpy as np
import argparse

### Main entry point

def main(args=None):
    
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--mode", choices=["sitl", "hardware"], default="sitl")

    # IMPORTANT: keep ROS args untouched
    cli_args, ros_args = parser.parse_known_args(args=args)

    # Now ROS can safely parse its own arguments
    rclpy.init(args=ros_args)

    ### Load config
    cfg_path = os.path.join(get_package_share_directory('px505_controller'), 'config.yaml')
    cfg = load_config(cfg_path)    

    ### Control laws and agents setup
    control_laws = {} # ignore # type: dict[int, ControlLaw] - map from controlled agent id to control law instance
    crazyflies = [] # ignore # type: list[CrazyflieDescriptor] - list of Crazyflie descriptors (id, uri, current state, etc.)

    env_controllers_config = cfg.controllers.sitl if cli_args.mode == "sitl" else cfg.controllers.hitl
    env_agents_config = cfg.agents.sitl if cli_args.mode == "sitl" else cfg.agents.hitl

    for ctrl_name, ctrl_cfg in env_controllers_config.controllers.items():
        control_law = __instantiate_control_law(ctrl_cfg)
        control_laws[ctrl_cfg.controlled_agent_id] = control_law

    for crazyflie_name, crazyflie_cfg in env_agents_config.quadrotors.crazyflies.items():
        crazyflie_desc = CrazyflieDescriptor(
            name=crazyflie_name,
            id=crazyflie_cfg.id,
            uri=crazyflie_cfg.uri,
            mass=crazyflie_cfg.mass
        )
        crazyflies.append(crazyflie_desc)

    controller = ControllerNode(crazyflies=crazyflies, control_laws=control_laws)

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down...")
    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def __instantiate_control_law(controllers_config: Controller) -> ControlLaw:
    """
    Factory function to instantiate a control law based on the controller configuration.
    This function checks the type of the controller and creates an instance of the corresponding control law.
    """
    control_law = None

    params = controllers_config.parameters
    dt = params.dt / 1000.0  # convert ms to seconds for control law settings

    match controllers_config.controller_type:
        case "PID":
            setting = PIDControlLawSetting(model=DoubleIntegrator3DOF.from_dt(dt))
            setting.set_gains(Kp=np.array(params.kp), Ki=np.array(params.ki), Kd=np.array(params.kd))
            if params.u_min is not None and params.u_max is not None:
                setting.set_saturation_limits(u_min=np.array(params.u_min), u_max=np.array(params.u_max))

            control_law = PIDControlLaw(setting)

        case "LQR":
            setting = LQRControlLawSetting(model=DoubleIntegrator3DOF.from_dt(dt))
            setting.set_weights(Q=np.diag(params.Q), R=np.diag(params.R))
            if params.u_min is not None and params.u_max is not None:
                setting.set_saturation_limits(u_min=np.array(params.u_min), u_max=np.array(params.u_max))

            control_law = LQRControlLaw(setting)

        case "MPC":
            setting = MPCControlLawSetting(model=DoubleIntegrator3DOF.from_dt(dt), Npred=params.horizon)
            setting.set_weights(Q=np.diag(params.Q), R=np.diag(params.R))
            if params.u_min is not None and params.u_max is not None:
                setting.set_input_constraints(u_min=np.array(params.u_min), u_max=np.array(params.u_max))
            if params.x_min is not None and params.x_max is not None:
                setting.set_state_constraints(x_min=np.array(params.x_min), x_max=np.array(params.x_max))

            control_law = MPCControlLaw(setting)

        case _:
            raise ValueError(f"Unsupported controller type: {controllers_config.controller_type}")

    return control_law

if __name__ == "__main__":
    main()

