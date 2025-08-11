import control
import numpy as np

from models.trajectory_generators import (
    generate_static_noise_trajectory,
    generate_trajectory_with_static_noise,
    StablizingNoiseTrajectory,
    CommandFollowingTrajectory
)
from models.inverted_pendulum_closed_loop import (
    create_stabilizing_plant,
    create_lqr_stabilizing_and_command_following_plant
)

def simulate_closed_loop_stabiling_plant(theta_init: float, v_init: float) -> control.TimeResponseData:

    stabilizing_plant = create_stabilizing_plant()
    static_noise_trajectory: StablizingNoiseTrajectory = generate_static_noise_trajectory(dt=0.01, t_Final=10, scale=0.1)
    print(stabilizing_plant)

    print("-----" * 30)
    print("Simulating Input Output response")
    print("-----"*30)
    response = control.input_output_response(
        stabilizing_plant,
        static_noise_trajectory.t,
        static_noise_trajectory.noise,
        initial_state=[0, v_init, theta_init, 0]
    )
    print("\nCompleted\n")
    print("-----" * 30)

    return response


def simulate_closed_loop_command_following_plant() -> control.TimeResponseData:

    stabilizing_plant = create_lqr_stabilizing_and_command_following_plant()
    trajectory: CommandFollowingTrajectory = generate_trajectory_with_static_noise(dt=0.01, t_final=20, scale=0.5)
    print("-----" * 30)
    print("Simulating Input Output response")
    print("-----" * 30)
    print(stabilizing_plant)
    print(stabilizing_plant.dynamics(0, np.ones(4), np.ones(8)))
    response = control.input_output_response(
        stabilizing_plant,
        trajectory.t,
        np.vstack([
            trajectory.noise,
            trajectory.x_d
        ]),
        initial_state = [0, 0, np.pi, 0]
    )
    print("-----" * 30)
    print("\nCompleted\n")
    print("-----" * 30)
    print(response.to_pandas().head().to_string())
    print("-----" * 30)
    return response
