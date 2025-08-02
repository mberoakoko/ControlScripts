import control
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from systems import PlantFactory
matplotlib.use("TkAgg")
plt.style.use("bmh")
plt.rcParams.update({"font.size": 8})

def plot_continuous_vs_discrete_plant(plant_factory: PlantFactory = PlantFactory(dt=0.01)) -> None :
    plant = plant_factory.create()
    discrete_plant = plant_factory.discretize()
    fig: Figure = plt.figure(figsize=(16//2, 9//2))
    ax: Axes = fig.add_subplot()
    t, y = control.step_response(plant, 0.1)
    t_discrete, y_discrete = control.step_response(discrete_plant, 0.1)
    ax.plot(t, y, linewidth=0.8, color="C1", label="continuous_time_function")
    ax.plot(t_discrete, y_discrete, ".-" ,linewidth=0.8, color="C2", label="discrete_time_function")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_continuous_vs_discrete_plant()