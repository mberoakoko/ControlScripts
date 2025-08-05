import control
import pandas as pd
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
    u_d = np.array([1 * np.ones_like(timepts), np.zeros_like(timepts)])
    return StaticTrajectory(
        t=timepts,
        x_d=x_d,
        u_d=u_d
    )

def simulate_lqr_system_dynamics() -> control.TimeResponseData:
    lqr_controlled_plant = VehiclePlant().create_closed_loop_system()
    trajectories = generate_static_trajectory(t_final=10, dt=0.01)
    return control.input_output_response(lqr_controlled_plant, trajectories.t,
                                         np.vstack((trajectories.x_d, trajectories.u_d)), 0)

if __name__ == "__main__":
    sim_results = simulate_lqr_system_dynamics()
    print(sim_results.state_labels)
    results_as_pandas: pd.DataFrame = sim_results.to_pandas()
    print(results_as_pandas.head())