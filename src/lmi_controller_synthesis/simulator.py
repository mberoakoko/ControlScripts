import control
import numpy as np

from model.models import MassSpringDamperExogenous
from controllers_and_observers.robust_controllers import LowerStarController
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
    raw_controller_full_state = LowerStarController(
        params=control.ss(
            np.zeros_like(raw_plant.A),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.ones((2, 2))
        )
    )
    plant = raw_plant.as_non_linear_io_system()
    ctrl_plant = raw_controller_full_state.as_non_linear_io_system()
    closed_Loop_system_ = control.interconnect([plant, ctrl_plant], inputs=["w",], outputs=["z", ])
    return closed_Loop_system_


if __name__ == "__main__":
    closed_Loop_system = create_lower_star_mass_spring_damper()
    print(closed_Loop_system)
    print(closed_Loop_system.dynamics(1, np.array([0, 0]), [0]))
