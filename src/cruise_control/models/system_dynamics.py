import control
import dataclasses
from math import copysign, sin

import numpy as np


def sign(x: float) -> float:
    return copysign(1, x)

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
        ...

    def vehicle_putput(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        ...

