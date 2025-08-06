import control
import dataclasses
import matplotlib.pyplot as plt

import numpy as np


@dataclasses.dataclass
class BycycleModel:
    l: float = dataclasses.field(default=3.0)           # Wheel base
    a: float = dataclasses.field(default=1.5)
    phi_max: float = dataclasses.field(default=0.5)     # max steering angle in radians

    def __vehicle_update(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        v, delta = u
        x, y , theta = x
        delta = np.clip(delta, -self.phi_max, self.phi_max)
        alpha = np.arctan2(self.a * np.tan(delta), self.l)
        return np.array([
            np.cos(theta + alpha) * v,  # xdot = cos(theta) v
            np.sin(theta + alpha) * v,  # ydot = sin(theta) v
            (v / self.a) * np.sin(alpha)  # thdot = v/l tan(phi)
        ])

    def __vehicle_output(self, t, x: np.ndarray, u: np.ndarray, parmas):
        return x

    def as_non_linear_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__vehicle_update, self.__vehicle_output,
            name="Nonlinear Vehicle Plant",
            states=["x", "y", "theta"],
            inputs=("v", "delta"),
            outputs=("x", "y", "theta")
        )

    def as_linear_stata_space(self, x_equilibrium: np.ndarray, u_equilibrium: np.ndarray) -> control.StateSpace:
        return control.linearize(self.as_non_linear_system(), x_equilibrium.tolist(), u_equilibrium.tolist())

# @dataclasses.dataclass
# class NoiseBlock:
#     scale: float = dataclasses.field(default=0.1)
#     dt: float = dataclasses.field(default=0.0001)
#     delay_steps: int = dataclasses.field(default=10)
#
#     def _noise_update(self, t, x: np.ndarray, u: np.ndarray, params):
#         x_new = np.roll(x, 1, axis=1)
#         x_new[:, 0] = u
#         return x_new
#
#     def __noise_outputs(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
#         delayed_x_signal = x[:, -1]
#         noise = np.random.normal(loc=0, scale=self.scale, size=delayed_x_signal.shape)
#         return delayed_x_signal + noise
#
#     def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
#         return control.NonlinearIOSystem(
#             self.__noise_outputs, self.__noise_outputs, name="NoiseBlock",
#             inputs=("x", "y", "theta"),
#             outputs=("x_n", "y_n", "theta_n"),
#             states=3 * self.delay_steps,
#             dt=self.dt
#         )

@dataclasses.dataclass
class NoiseBlock:
    scale: float = dataclasses.field(default=0.1)

    def __noise_outputs(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        noise = np.random.normal(loc=0, scale=self.scale, size=u.shape)
        return u + noise

    def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, self.__noise_outputs, name="NoiseBlock",
            inputs=("x", "y", "theta"),
            outputs=("x_n_1", "y_n_1", "theta_n_1"),
        )

@dataclasses.dataclass
class DelayBlock:
    name: str = dataclasses.field(default="DelayBlock")

    @staticmethod
    def __delay_update(t, x: np.ndarray, u: np.ndarray, params):
        return  u

    @staticmethod
    def __noise_outputs(t, x: np.ndarray, u:np.ndarray, params):
        return x

    def create_delay_block(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            DelayBlock.__delay_update, DelayBlock.__noise_outputs,
            name=self.name,
            states=("x_n_1", "y_n_1", "theta_n_1"),
            inputs=("x_n_1", "y_n_1", "theta_n_1"),
            outputs=("x_n", "y_n", "theta_n"),
        )


@dataclasses.dataclass
class NoiseAndDelayBlock:
    scale: float = dataclasses.field(default=0.1)

    def __update(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        noise = np.random.normal(loc=0, scale=self.scale, size=u.shape)
        return u + noise

    def __output(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        return x

    def as_non_linear_io_system(self, num_states: int) -> control.NonlinearIOSystem:

        return control.NonlinearIOSystem(
            self.__update,
            self.__output,
            states=num_states,
            name="NoiseAndDelayBlock",
            inputs=("x", "y", "theta"),  # The input from the plant
            outputs=("x_n", "y_n", "theta_n"),  # The delayed noisy output
        )

def initial_test():
    test_model = BycycleModel().as_non_linear_system()
    # print(test_model)
    vehicle_state_space = BycycleModel().as_linear_stata_space(
        x_equilibrium=np.array([10, 0, 0]),
        u_equilibrium=np.array([10, 0])
    )
    # print(np.linalg.eigvals(vehicle_state_space.A))
    # print(np.linalg.matrix_rank(control.ctrb(vehicle_state_space.A, vehicle_state_space.B)))
    noise_block = NoiseBlock().as_non_linear_io_system()
    noisy_plant = control.interconnect(
        [test_model, noise_block],
        inputs=test_model.input_labels,
        outputs=noise_block.output_labels
    )

    print(noisy_plant)
    print("\n")
    print(test_model.dynamics(0, np.ones(3), np.ones(2)))
    noisy_plant.dynamics(0, np.ones(3), np.ones(2))


def perform_test_simulation_for_noisy_plant() -> None:
    test_noise_block = NoiseBlock().as_non_linear_io_system()
    t_final = 1
    n = round(t_final/0.01)
    print(f"{n=}")
    t = np.linspace(0, t_final, n)
    x_test = np.array([
        np.ones_like(t),
        np.ones_like(t),
        np.ones_like(t)
    ])

    test_plant = BycycleModel().as_non_linear_system()
    delay_block = DelayBlock().create_delay_block()
    # noise_and_dalay_block = control.interconnect(
    #     [test_noise_block, delay_block],
    #     inputs=test_noise_block.input_labels,
    #     outputs=delay_block.output_labels
    # )
    noise_and_delay_block = NoiseAndDelayBlock(scale=0.1).as_non_linear_io_system(num_states=3)
    test_noisy_plant = control.interconnect(
        [test_plant, noise_and_delay_block],
        inputs=test_plant.input_labels,
        outputs=noise_and_delay_block.output_labels
    )
    print(test_noisy_plant)

    result = control.input_output_response(
        noise_and_delay_block, t, x_test, 0
    )
    result.to_pandas().plot.line()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    perform_test_simulation_for_noisy_plant()

