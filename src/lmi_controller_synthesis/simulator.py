import control
import numpy as np

from model.models import MassSpringDamperExogenous, DelayBlock
from controllers_and_observers.robust_controllers import LowerStarController, ControllerType
from controllers_and_observers.controller_synthesis import FullStateOptimalController, NineMatrixData
_SIMULATION_DT: float = 0.01

def create_lower_star_mass_spring_damper() -> control.NonlinearIOSystem:
    raw_plant = MassSpringDamperExogenous(
        m=1,
        c=1,
        k=1,
        alpha_1=1,
        alpha_2=1,
    )
    print(raw_plant.A.shape)
    controller_sythesizer = FullStateOptimalController(
        params=NineMatrixData(
            A=raw_plant.A,
            B_1=raw_plant.B_1,
            B_2=raw_plant.B_2,
            C_1=raw_plant.C_1,
            C_2=raw_plant.C_2,
            D_1_1=raw_plant.D_1_1,
            D_1_2=raw_plant.D_1_2,
            D_2_1=raw_plant.D_2_1,
            D_2_2=raw_plant.D_2_2,
        )
    )
    F = controller_sythesizer.f_matrix()
    print(F)
    raw_controller_full_state = LowerStarController(
        params=control.ss(
            np.zeros_like(raw_plant.A),
            np.zeros((2, 2)),
            np.zeros((1, 2)),
            F
        )
    )
    plant = raw_plant.as_non_linear_io_system()
    ctrl_plant = raw_controller_full_state.as_non_linear_io_system(controller_type=ControllerType.FULL_STATE_CONTROLLER)
    delay_block = DelayBlock().as_nonlinear_io_system()
    u_delay = control.TransferFunction(*control.pade(0.01, 7), inputs=["u_prime"], outputs=["u"]).to_ss()
    print(f"{u_delay=}")
    closed_Loop_system_ = control.interconnect([plant, delay_block, ctrl_plant, u_delay], inputs=["w",], outputs=["z", ])
    return closed_Loop_system_


def try_out_closed_loop_system_with_mock_controller():
    closed_Loop_system = create_lower_star_mass_spring_damper()

    print(closed_Loop_system)

def try_out_fullstate_controller_synthesis():
    raw_plant = MassSpringDamperExogenous(
        m=10,
        c=1,
        k=12,
        alpha_1=1,
        alpha_2=1,
    )
    print(raw_plant.A.shape)
    controller_sythesizer = FullStateOptimalController(
        params=NineMatrixData(
            A=raw_plant.A,
            B_1=raw_plant.B_1,
            B_2=raw_plant.B_2,
            C_1=raw_plant.C_1,
            C_2=raw_plant.C_2,
            D_1_1=raw_plant.D_1_1,
            D_1_2=raw_plant.D_1_2,
            D_2_1=raw_plant.D_2_1,
            D_2_2=raw_plant.D_2_2,
        )
    )

    print(controller_sythesizer)
    controller_sythesizer.f_matrix()



def simulate_mass_spring_damper(dt=_SIMULATION_DT, t_final=10):
    system = create_lower_star_mass_spring_damper()
    print(system)
    t_sim = np.linspace(0, t_final, int(t_final/dt))
    u = np.zeros_like(t_sim)
    u[t_sim < 2] = 0
    response: control.TimeResponseData = control.input_output_response(system, t_sim, u)


if __name__ == "__main__":
    simulate_mass_spring_damper()