import numpy as np
from typing import NamedTuple


class StablizingNoiseTrajectory(NamedTuple):
    t: np.ndarray
    noise: np.ndarray

def generate_static_noise_trajectory(dt: float, t_Final, scale: float = 0.1) -> StablizingNoiseTrajectory:
    time = np.linspace(0, t_Final, round(t_Final/dt))
    return StablizingNoiseTrajectory(
        t = time,
        noise=np.random.normal(loc=0, scale=scale, size=(4, time.shape[0]))
    )