import control
import numpy as np

from model.models import MassSpringDamperExogenous
from controllers_and_observers.robust_controllers import LowerStarController
from controllers_and_observers.controller_synthesis import FullStateOptimalController, NineMatrixData
_SIMULATION_DT: float = 0.001

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
    ctrl_plant = raw_controller_full_state.as_non_linear_io_system()
    closed_Loop_system_ = control.interconnect([plant, ctrl_plant], inputs=["w",], outputs=["z", ])
    return closed_Loop_system_


def try_out_closed_loop_system_with_mock_controller():
    closed_Loop_system = create_lower_star_mass_spring_damper()
    print(closed_Loop_system)
    print(closed_Loop_system.dynamics(1, np.array([0, 0]), [0]))
    print(closed_Loop_system.output(1, np.array([0, 0]), [0]))


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


if __name__ == "__main__":
    create_lower_star_mass_spring_damper()