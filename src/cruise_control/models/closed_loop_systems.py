import control
import numpy as np
from typing import NamedTuple

from controller import StateSpaceIntegralController, generate_controller_gains, ControllerGains
from system_dynamics import VehicleDynamics


class TimeAndInputTrajectories(NamedTuple):
    t: np.ndarray
    v_ref: np.ndarray
    gear: np.ndarray
    theta_0: np.ndarray


def create_trajectories(dt: float, t_final: float) -> TimeAndInputTrajectories:
    t = np.linspace(0, t_final, round(t_final/dt))
    return TimeAndInputTrajectories(
        t       = t,
        v_ref   = 20 * np.ones(t.shape),
        gear    = 4 * np.ones(t.shape),
        theta_0 = np.zeros(t.shape)
    )

def create_hilly_trajectory(flat_trajectory: TimeAndInputTrajectories):
    T = flat_trajectory.t
    theta_hill = [
        0 if t <= 5 else
        4. / 180. * np.pi * (t - 5) if t <= 6 else
        4. / 180. * np.pi for t in T]
    return TimeAndInputTrajectories(
        t = flat_trajectory.t,
        v_ref=flat_trajectory.v_ref,
        gear=flat_trajectory.gear,
        theta_0=np.array(theta_hill)
    )

def find_equilibrium(vehicle_plant: control.InputOutputSystem, v_ref: float, gear: float, theta: float):
    return control.find_eqpt(vehicle_plant, [v_ref], [0, gear, theta], y0=[v_ref], iu=[1, 2])

def create_closed_loop_system() -> control.InputOutputSystem:
    vehicle_dynamics  = VehicleDynamics()
    vehicle_plant = vehicle_dynamics.as_non_linear_io_system()
    x_eq, u_eq = find_equilibrium(vehicle_plant, 20, 4, 0)
    print(f"X equilibrium {x_eq}")
    print(f"u equilibrium {u_eq}")
    linearized_vehicle_plant = control.linearize(vehicle_plant, x_eq, [u_eq[0],4, 0])
    print(linearized_vehicle_plant)
    controller_gains = generate_controller_gains(linearized_vehicle_plant)
    print(controller_gains)


if __name__ == "__main__":
    print(create_trajectories(dt=0.01, t_final=25))
    create_closed_loop_system()