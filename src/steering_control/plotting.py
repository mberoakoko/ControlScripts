import control
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from simulations import simulate_lqr_system_dynamics, simulate_lqr_system_noisy_dynamics, simulate_lqr_system_with_exogenous_noise

matplotlib.use("TkAgg")
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
plt.rcParams.update({"font.size":7})

def plot_lqr_controller_response(time_response: control.TimeResponseData) -> None:
    time_response_as_df = time_response.to_pandas()
    print(time_response_as_df.columns)
    fig: Figure = plt.figure(figsize=(16/1.2, 9//1.5))
    ax_1: Axes = fig.add_subplot(611)
    ax_2: Axes = fig.add_subplot(612)
    ax_3: Axes = fig.add_subplot(613)
    ax_4: Axes = fig.add_subplot(614)
    ax_5: Axes = fig.add_subplot(615)
    ax_6: Axes = fig.add_subplot(616)

    ax_1.plot(time_response_as_df["time"], time_response_as_df["x_d"],"--", linewidth=0.7, color="C3", label="Desired x_d")
    ax_1.plot(time_response_as_df["time"], time_response_as_df["x"], color="C3", label="x")

    ax_2.plot(time_response_as_df["time"], time_response_as_df["y_d"],"--", linewidth=0.7, color="C4", label="Desired y_d")
    ax_2.plot(time_response_as_df["time"], time_response_as_df["y"], color="C4", label="y")

    ax_3.plot(time_response_as_df["time"], time_response_as_df["theta_d"],"--", linewidth=0.7, color="C5", label="Desired theta_d")
    ax_3.plot(time_response_as_df["time"], time_response_as_df["theta"], color="C5", label="theta")

    ax_4.plot(time_response_as_df["time"], time_response_as_df["delta"], color="C6", label="delta")
    ax_4.axhline(y=1.5, color='C4', lw=0.5, ls='--')
    ax_4.axhline(y=-1.5, color='C4', lw=0.5, ls='--')


    ax_5.plot(time_response_as_df["x_d"], time_response_as_df["y_d"], "--", color="C1",
              label="DesiredTrajectory", linewidth=0.7)

    ax_5.plot(time_response_as_df["x"], time_response_as_df["y"], "-", color="C1",
              label="DesiredTrajectory", linewidth=0.9)
    desired_velocity = np.diff(time_response_as_df["x_d"])/np.diff(time_response_as_df["time"])
    plant_velocity = np.diff(time_response_as_df["x"])/np.diff(time_response_as_df["time"])
    ax_6.plot(time_response_as_df["x_d"][:-1], desired_velocity,
              "--", linewidth=0.8,  color="C3", label="Desired Velocity")

    ax_6.plot(time_response_as_df["x"][:-1], plant_velocity,
              "-", color="C3", label="Desired Velocity")

    for item in [ax_1, ax_2, ax_3, ax_4, ax_5]:
        item.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    plot_lqr_controller_response(
        time_response=simulate_lqr_system_with_exogenous_noise()
    )

