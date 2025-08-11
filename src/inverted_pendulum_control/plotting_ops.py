import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import control

matplotlib.use("TkAgg")
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
plt.rcParams.update({"font.size":7})

def add_legend_and_plot(ax_1, ax_2, ax_3, ax_4, ax_5, data, t_data):
    ax_5.plot(t_data, data["u_force"], linewidth=0.7, color="C5", label="u_force")
    for axis in [ax_1, ax_2, ax_3, ax_4, ax_5]:
        axis.legend()
    plt.title("Stabiling Controller")
    plt.tight_layout()
    plt.show()


def simple_stabilizing_plot(response_data: control.TimeResponseData) -> None:
    data = response_data.to_pandas()
    t_data = data["time"]
    fig: Figure = plt.figure(figsize=(16//2, 9//2))
    ax_1: Axes = fig.add_subplot(511)
    ax_2: Axes = fig.add_subplot(512)
    ax_3: Axes = fig.add_subplot(513)
    ax_4: Axes = fig.add_subplot(514)
    ax_5: Axes = fig.add_subplot(515)
    ax_1.plot(t_data, data["x"], linewidth=0.7, color="C1", label="x")
    ax_2.plot(t_data, data["v"], linewidth=0.7, color="C2", label="v")
    ax_3.plot(t_data, data["theta"], linewidth=0.7, color="C3", label="theta")
    ax_4.plot(t_data, data["theta_dot"], linewidth=0.7, color="C4", label="theta_dot")
    ax_5.plot(t_data, data["u_force"], linewidth=0.7, color="C5", label="u_force")
    for axis in [ax_1, ax_2, ax_3, ax_4, ax_5]:
        axis.legend()
    plt.title("Stabiling Controller")
    plt.tight_layout()
    plt.show()