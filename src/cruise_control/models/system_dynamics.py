import control
import dataclasses
from math import copysign, sin
from typing import Callable

import numpy as np


def sign(x: float | np.ndarray) -> float:
    return copysign(1, x)

@dataclasses.dataclass
class MotorTorqueFunctor:
    tm: float = dataclasses.field(default=190)
    omega_m: float = dataclasses.field(default=430)
    beta: float = dataclasses.field(default=0.5)

    def __call__(self, omega: float | np.ndarray) -> np.ndarray:
        return np.clip(self.tm * (1 - self.beta * (omega/self.omega_m - 1)**2), 0, None)

motor_torque = MotorTorqueFunctor()

@dataclasses.dataclass
class VehicleDynamics:
    m: float            = dataclasses.field(default=1600)
    g: float            = dataclasses.field(default=9.8)
    c_r: float          = dataclasses.field(default=0.01)
    c_d: float          = dataclasses.field(default=0.32)
    rho: float          = dataclasses.field(default=1.3)
    A: float            = dataclasses.field(default=2.4)  # car area
    alpha: list[float]  = dataclasses.field(default=(40, 25, 16, 12, 10))
    # motor_torque: Callable[[float], np.ndarray] = dataclasses.field(default=MotorTorqueFunctor())

    def vehicle_update(self, t, x:np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        throttle, gear, theta = u
        velocity = x
        throttle = np.clip(throttle, 0, 1)
        omega = self.alpha[int(gear) - 1] * velocity
        f = self.alpha[int(gear) - 1] * motor_torque(omega) * throttle

        f_g = self.m * self.g * sin(theta)
        f_t = self.m * self.g * self.c_r * sign(velocity)
        f_a = 1/2 * self.rho * self.c_d * self.A * abs(velocity) * velocity
        f_disturbance = f_g + f_t + f_a
        return ( f - f_disturbance)/self.m


    def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.vehicle_update, None, name="Vehicle",
            inputs=("u", "gear", "theta"), outputs=("v",), states=("v",)
        )


if __name__ == "__main__":
    motor_torque = MotorTorqueFunctor()
    t_omegas = np.arange(0, 100, 1)
    print(motor_torque(omega=1))
    print(np.vectorize(motor_torque)(t_omegas))