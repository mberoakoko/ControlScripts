import control
import dataclasses
from math import copysign, sin

import numpy as np


def sign(x: float) -> float:
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
    m: float
    g: float
    c_r: float
    c_d: float
    rho: float
    A: float
    alpha: float

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

    def vehicle_putput(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        ...

@dataclasses.dataclass
class MotorTorqueFunctor:
    tm: float = dataclasses.field(default=190)
    omega_m: float = dataclasses.field(default=430)
    beta: float = dataclasses.field(default=0.5)
