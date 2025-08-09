import control
import dataclasses

import numpy as np
from scipy import signal

def butter_filter_factory(taps, nyquist: float = 0.5 * 100 , cuttoff: float = 5):
    result = signal.butter(taps, cuttoff / nyquist, btype="Low", analog=False, output="ba")
    b_, a_ = result
    return b_, a_


@dataclasses.dataclass
class FilterChannel:
    a: list[float] = dataclasses.field()
    b: list[float] = dataclasses.field()
    z_i: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.z_i = signal.lfilter_zi(self.b, self.a)

    def __call__(self, t, x: np.ndarray, u: np.ndarray, params = None):
        out, new_z_i = signal.lfilter(self.b, self.a, x, zi=self.z_i)
        return new_z_i.squeeze(), out.squeeze()

@dataclasses.dataclass
class FilterBlock:
    x_channel_filter: FilterChannel = FilterChannel(*butter_filter_factory(taps=BLOCK_TAPS))
    v_channel_filter: FilterChannel = FilterChannel(*butter_filter_factory(taps=BLOCK_TAPS))
    theta_channel_filter: FilterChannel =  FilterChannel(*butter_filter_factory(taps=BLOCK_TAPS))
    theta_dot_channel_filter: FilterChannel = FilterChannel(*butter_filter_factory(taps=BLOCK_TAPS))


    def __update_block_state(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        x_val, v_val, theta_val, theta_dot_val = u
        x_filtered = self.x_channel_filter(t, np.array([x_val]), None, None),
        v_Filtered = self.v_channel_filter(t, np.array([v_val]), None, None),
        theta_filtered = self.theta_channel_filter(t, np.array([theta_val]), None, None)
        theta_dot_filtered = self.theta_dot_channel_filter(t, np.array([theta_dot_val]), None, None)
        print(f"{theta_dot_filtered=}")
        return np.array([
            x_filtered[0],
            v_Filtered[0],
            theta_filtered,
            theta_dot_filtered
        ])

    def __output_block_state(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        return x

    def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__update_block_state, self.__output_block_state,
            name="FilterBlock",
            states=["x_hat", "v_hat", "theta_hat", "theta_dot_hat"],
            inputs=["x", "v", "theta", "theta_dot"],
            outputs=["x_hat", "v_hat", "theta_hat", "theta_dot_hat"],
        )

if __name__ == "__main__":
    b, a = butter_filter_factory(taps=10)
    channel_filter = FilterChannel(
        a = a,
        b = b
    )
    print(channel_filter.z_i)
    new_zi_, out_ = channel_filter(0, np.array([1]), channel_filter.z_i)
    print(f"{new_zi_=}")
    print(f"{out_=}")