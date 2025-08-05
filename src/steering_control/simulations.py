import control
import numpy
from typing import NamedTuple

import numpy as np

from models.closed_loop_plants import VehiclePlant

class StaticTrajectory(NamedTuple):
    t: np.ndarray
    x_d: np.ndarray
    u_d: np.ndarray

def generate_static_trajectory(t_final: float, dt: float) -> StaticTrajectory:
    timepts = np.linspace(0, t_final, round(t_final/dt))
    x_d = np.array([
        10 * timepts + 2 * (timepts - 5) * (timepts > 5),
        0.5 * np.sin(timepts * 2 * np.pi),
        np.zeros_like(timepts)
    ])
    u_d = np.array([10 * np.ones_like(timepts), np.zeros_like(timepts)])
    return StaticTrajectory(
        t=timepts,
        x_d=x_d,
        u_d=u_d
    )

def simulate_lqr_system_dynamics() -> control.TimeResponseData:
    lqr_controlled_plant = VehiclePlant().create_closed_loop_system()
    
    return control.input_output_response(lqr_controlled_plant, )