import numpy as np
from simulator import simulate_closed_loop_stabiling_plant
from plotting_ops import simple_stabilizing_plot

if __name__ == "__main__":
    response = simulate_closed_loop_stabiling_plant(
        theta_init=np.pi + 0.01,
        v_init=-1
    )
    print(response.to_pandas().columns)
    print(response.to_pandas().head())
    print(response.to_pandas().info())
    simple_stabilizing_plot(
        response_data=response
    )