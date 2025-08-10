import control
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


matplotlib.use("TkAgg")
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
plt.rcParams.update({"font.size":7})

def plot_mech_system_response(respose_data: control.TimeResponseData):
    data: pd.DataFrame = respose_data.to_pandas()
    t_sim = data.iloc[0, :]
    tau = data["tau"]
    theta, theta_dot = data["theta"], data["theta_dot"]
    fig: Figure = plt.figure(figsize=(19//2, 9//2))
    ax_1: Axes = fig.add_subplot(311)
    ax_2: Axes = fig.add_subplot(312)
    ax_3: Axes = fig.add_subplot(313)
    ax_1.plot( theta, color="C1", label="theta")
    ax_2.plot( theta_dot, color="C2", label="theta_dot")
    ax_3.plot( tau, color="C3", label="tau")

    for item in [ax_1, ax_2, ax_3]:
        item.legend()

    plt.tight_layout()
    plt.show()
