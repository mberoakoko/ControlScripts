import control
from typing import NamedTuple
import numpy as np


class StateSpaceIntegralController:
    k:   float | np.ndarray
    k_i: float | np.ndarray
    k_f: float | np.ndarray
    x_d: float          # reference state
    y_d: float          # reference point
    u_d: float          # desired forcing


    def integral_controller_update(self, t, z: float, u: np.ndarray, params: dict) -> float:
        y, r = u
        return y - r

    def integral_controller_output(self, t, z: float, u: np.ndarray, params):
        x, y, r = u
        return self.u_d + self.k * (x - self.x_d) - self.k_i * z + self.k_f * (r - self.y_d)

    def generate(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.integral_controller_update, self.integral_controller_output,
            inputs=("x", "y", "r"),
            outputs=("u", ),
            states=("z",)
        )

class ControllerGains(NamedTuple):
    K: float| np.ndarray
    K_I: float | np.ndarray
    K_F: float | np.ndarray

def generate_controller_gains(lineaerized_system: control.StateSpace) -> ControllerGains:
    A: np.ndarray = lineaerized_system.A
    B: np.ndarray = lineaerized_system.B
    C: np.ndarray = lineaerized_system.C
    K = 0.4
    return ControllerGains(
        K = K,
        K_I = 0.1,
        K_F = -1/(C * np.linalg.inv(A - B @ K) * B)
    )