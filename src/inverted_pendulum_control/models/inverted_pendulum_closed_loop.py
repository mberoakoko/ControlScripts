import control

from .inverted_pendulum_model import InvertedPendulum, linearize_plant
from .controllers import LQR_Stabilizing_Controller, LQR_CommandFollowind_Controller

def create_stabilizing_plant() -> control.NonlinearIOSystem:
    plant_factory = InvertedPendulum()
    plant_linearized = linearize_plant(plant_factory)
    pendulum_controller = LQR_Stabilizing_Controller(
        plant=plant_linearized
    ).as_non_linear_output_function()
    closed_loop_plant = control.interconnect(
        [plant_factory.as_non_linear_io_system_full_state_measurement(), pendulum_controller],
        name="ClosedLoopInvertedPendulumFullStatecontrol",
        inputs=["x_n", "v_n", "theta_n", "theta_dot_n"], # Inputs: Noise
        outputs=["x", "v", "theta", "theta_dot", "u_force"] # Outputs: Full State and forcing
    )
    return closed_loop_plant


def create_lqr_stabilizing_and_command_following_plant() -> control.NonlinearIOSystem:
    """
    This function creates a lqr stability full state command following plant
    :return:
    """
    plant_factory = InvertedPendulum()
    plant_linearized = linearize_plant(plant_factory)
    pendulum_controller = LQR_CommandFollowind_Controller(
        plant=plant_linearized
    ).as_non_linear_output_function()
    closed_loop_plant = control.interconnect(
        [plant_factory.as_non_linear_io_system_full_state_measurement(), pendulum_controller],
        name="ClosedLoopInvertedPendulumFullStatecontrol",
        inputs=["x_n", "v_n", "theta_n", "theta_dot_n", "x_d", "v_d", "theta_d", "theta_dot_d"], # Inputs: Noise
        outputs=["x", "v", "theta", "theta_dot", "u_force"] # Outputs: Full State and forcing
    )
    return closed_loop_plant
