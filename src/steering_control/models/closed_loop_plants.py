import dataclasses
import numpy as np
import control

from .system_dynamics import BycycleModel
from .controllers import LQRController


@dataclasses.dataclass
class VehiclePlant:
    plant: BycycleModel = dataclasses.field(default=BycycleModel())
    controller: control.NonlinearIOSystem = dataclasses.field(init=False)

    def __post_init__(self):
        self.controller = LQRController(
            linearized_plant=self.plant.as_linear_stata_space(
                x_equilibrium=np.array([10, 0, 0]),
                u_equilibrium=np.array([10, 0])
            )
        ).as_non_linear_io_system()

    def create_closed_loop_system(self) -> control.NonlinearIOSystem:
        return control.interconnect(
            [self.plant.as_non_linear_system(), self.controller],
            states=["x", "y", "theta"],
            inputs=["x_d","y_d", "theta_d", "v_d", "delta_d",],
            outputs=['x', 'y', 'theta', "delta"]
        )




if __name__ == "__main__":
    print(VehiclePlant(
        plant=BycycleModel()
    ).create_closed_loop_system())