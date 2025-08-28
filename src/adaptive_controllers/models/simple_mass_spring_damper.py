from email.policy import default
from sys import deactivate_stack_trampoline

import control
import dataclasses

import numpy as np


@dataclasses.dataclass
class SimpleMassSpringDamper:
    m: float = dataclasses.field(default=5.0)
    alpha: float = dataclasses.field(default=0.1)
    beta: float = dataclasses.field(default=0.1)

    def __plant_update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        x_1, x_2 = x
        return np.array([
            x_2,
            (1/self.m) * (u - self.alpha * x_1 - self.beta * x_2**2)
        ])

    def __plant_output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return x

    def as_non_linear_plant(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__plant_update, self.__plant_output,
            name="SimpleMassSpringDamper",
            inputs=["u"],
            outputs=["x", "x_dot"]
        )