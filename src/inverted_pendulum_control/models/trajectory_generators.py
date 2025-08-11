import numpy as np
from typing import NamedTuple


class StablizingNoiseTrajectory(NamedTuple):
    t: np.ndarray
    noise: np.ndarray


class CommandFollowingTrajectory(NamedTuple):
    t: np.ndarray
    x_d: np.ndarray
    noise: np.ndarray


def generate_static_noise_trajectory(dt: float, t_Final, scale: float = 0.1) -> StablizingNoiseTrajectory:
    time = np.linspace(0, t_Final, round(t_Final/dt))
    return StablizingNoiseTrajectory(
        t = time,
        noise=np.random.normal(loc=0, scale=scale, size=(4, time.shape[0]))
    )

def generate_trajectory_with_static_noise(dt: float, t_final: float, scale = 0.1) -> CommandFollowingTrajectory:
    time = np.linspace(0, t_final, round(t_final / dt))
    x_desired = (np.zeros_like(time) + (2 * np.ones_like(time) * (time > 4))
                                    +  (-6 * np.ones_like(time) * (time > 10))
                                    + ( 4 * np.ones_like(time) * (time > 16)))
    x_d = np.vstack([
        x_desired,
        np.zeros_like(time),
        np.pi*np.ones_like(time),
        np.zeros_like(time),
    ])
    return CommandFollowingTrajectory(
        t = time,
        x_d=x_d,
        noise=np.random.normal(loc=0, scale=scale, size=(4, time.shape[0]))
    )
