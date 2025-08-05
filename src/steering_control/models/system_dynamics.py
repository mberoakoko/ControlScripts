import control
import dataclasses

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
            self.__vehicle_update, self.__vehicle_output, states=3, name="Nonlinear Vehicle Plant",
            inputs=("v", "delta"),
            outputs=("x", "y", "theta")
        )

    def as_linear_stata_space(self, x_equilibrium: np.ndarray, u_equilibrium: np.ndarray) -> control.StateSpace:
        return control.linearize(self.as_non_linear_system(), x_equilibrium.tolist(), u_equilibrium.tolist())


if __name__ == "__main__":
    test_model = BycycleModel().as_non_linear_system()
    print(test_model)
    vehicle_state_space = BycycleModel().as_linear_stata_space(
        x_equilibrium=np.array([10, 0, 0]),
        u_equilibrium=np.array([10 , 0])
    )
    print(np.linalg.eigvals(vehicle_state_space.A))
    print(np.linalg.matrix_rank(control.ctrb(vehicle_state_space.A, vehicle_state_space.B)))

