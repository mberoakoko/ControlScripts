import control
import numpy as np
import pathlib

import pandas as pd

from simulator import simulate_closed_loop_stabiling_plant
from plotting_ops import simple_stabilizing_plot
from observers.least_squares_filter import LeastSquaresFilter
from observers.linear_time_invariant_filtering import FilterBlock
from models.inverted_pendulum_closed_loop import InvertedPendulum, linearize_plant


def run_stabilization_control_algorithm(save_tag: str = "stabilization_simulation") -> None:
    save_location = pathlib.Path(__file__).parent / f"data/{save_tag}.csv"
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
    response.to_pandas().to_csv(save_location)

def test_least_squares_filter() -> None:

    def _load_data() -> pd.DataFrame:
        save_location = pathlib.Path(__file__).parent / "data/stabilization_simulation.csv"
        return pd.read_csv(save_location)

    test_inverted_pendulum = InvertedPendulum()
    pend_linarized = linearize_plant(test_inverted_pendulum)
    filter_obj = LeastSquaresFilter(
        plant_discr=pend_linarized,
        with_tracing=True
    )
    filter = filter_obj.as_non_linear_io_system()
    print(filter)
    print("\n")

    test_x = np.ones(4)
    filter.dynamics(0, test_x, test_x, None)
    data_raw = _load_data()
    columns = ["x", "v", "theta", "theta_dot"]
    t = data_raw["time"].to_numpy()
    data: np.ndarray = data_raw.loc[:, columns].to_numpy().T
    filter_result = control.input_output_response(filter, t,  data, initial_state=data[:, 0])
    print(filter_result.to_pandas().head())
    filter_result.to_pandas().to_csv(pathlib.Path(__file__).parent / "data/test_least_squares_filter.csv")
    print("P trace")
    print(filter_obj.P_trace)

def test_lti_butter_worthfilter():
    def _load_data() -> pd.DataFrame:
        save_location = pathlib.Path(__file__).parent / "data/stabilization_simulation.csv"
        return pd.read_csv(save_location)

    filter_obj = FilterBlock()
    lti_filter = filter_obj.as_non_linear_io_system()
    data_raw = _load_data()
    columns = ["x", "v", "theta", "theta_dot"]
    t = data_raw["time"].to_numpy()[:20]
    data: np.ndarray = data_raw.loc[:, columns].to_numpy().T[:, :20]
    filter_result = control.input_output_response(lti_filter, t, data, initial_state=None)
    print(filter_result.to_pandas().to_string())

if __name__ == "__main__":
    test_lti_butter_worthfilter()