import dataclasses
import numpy as np
import control

from .system_dynamics import BycycleModel, NoiseBlock, NoiseAndDelayBlock
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

@dataclasses.dataclass
class VehiclePlantNoisy:
    plant: BycycleModel = dataclasses.field(default=BycycleModel())
    controller: control.NonlinearIOSystem = dataclasses.field(init=False)

    def __post_init__(self):
        self.controller = LQRController(
            linearized_plant=self.plant.as_linear_stata_space(
                x_equilibrium=np.array([10, 0, 0]),
                u_equilibrium=np.array([10, 0])
            )
        ).as_non_linear_io_system_noisy_block()

    def create_closed_loop_system(self) -> control.NonlinearIOSystem:
        nominal_plant = self.plant.as_non_linear_system()
        noise_block = NoiseAndDelayBlock().as_non_linear_io_system(num_states=3)
        print(f"{nominal_plant.state_labels=}")
        noise_plant = control.interconnect(
            [nominal_plant, noise_block],
            # states=nominal_plant.state_labels,
            inputs=nominal_plant.input_labels,
            outputs=noise_block.output_labels
        )
        print(noise_plant)
        print("---"*20)
        print(self.controller)
        return control.interconnect(
            [noise_plant, self.controller],
            states=["x", "y", "theta","x_n", "y_n", "theta_n"],
            inputs=["x_d","y_d", "theta_d", "v_d", "delta_d",],
            outputs=['x_n', 'y_n', 'theta_n', "delta"]
        )


if __name__ == "__main__":
    noisy_vehicle_plant = VehiclePlantNoisy()
    print(noisy_vehicle_plant.create_closed_loop_system())