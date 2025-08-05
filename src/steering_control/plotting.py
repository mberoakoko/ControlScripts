import control
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from simulations import simulate_lqr_system_dynamics

matplotlib.use("TkAgg")
plt.style.use("bmh")
plt.rcParams.update({"font.size":7})

def plot_lqr_controller_response(time_response: control.TimeResponseData) -> None:
    time_response_as_df = time_response.to_pandas()
    print(time_response_as_df.columns)
    fig: Figure = plt.figure(figsize=(16//2, 9//2))
    ax_1: Axes = fig.add_subplot(211)
    ax_2: Axes = fig.add_subplot(212)
    ax_1.plot(time_response_as_df["x_d"], time_response_as_df["y_d"], "--", color="C1",
              label="DesiredTrajectory", linewidth=0.7)

    ax_1.plot(time_response_as_df["x"], time_response_as_df["y"], "-", color="C1",
              label="DesiredTrajectory", linewidth=0.9)
    desired_velocity = np.diff(time_response_as_df["x_d"])/np.diff(time_response_as_df["time"])
    plant_velocity = np.diff(time_response_as_df["x"])/np.diff(time_response_as_df["time"])
    ax_2.plot(time_response_as_df["x_d"][:-1], desired_velocity,
              "--", linewidth=0.8,  color="C3", label="Desired Velocity")

    ax_2.plot(time_response_as_df["x"][:-1], plant_velocity,
              "-", color="C3", label="Desired Velocity")
    ax_1.legend()
    ax_2.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    plot_lqr_controller_response(
        time_response=simulate_lqr_system_dynamics()
    )

