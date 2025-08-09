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
    response.to_pandas().to_csv(save_location)

def test_least_squares_filter() -> None:

    def _load_data() -> pd.DataFrame:
        save_location = pathlib.Path(__file__).parent / "data/stabilization_simulation.csv"
        return pd.read_csv(save_location)

    test_inverted_pendulum = InvertedPendulum()
    pend_linarized = linearize_plant(test_inverted_pendulum)
    filter_obj = LeastSquaresFilter(
        plant_discr=pend_linarized
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


if __name__ == "__main__":
    test_least_squares_filter()