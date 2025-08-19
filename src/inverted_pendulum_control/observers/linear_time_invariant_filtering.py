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
        self.z_i = np.zeros_like(self.z_i)

    def __call__(self, t, x: np.ndarray, u: np.ndarray | None, params = None):
        out, self.z_i = signal.lfilter(self.b, self.a, x, zi=self.z_i)
        return out.squeeze()

BLOCK_TAPS: int = 4

@dataclasses.dataclass
class FilterBlock:
    x_channel_filter: FilterChannel = dataclasses.field(default_factory= lambda : FilterChannel(*butter_filter_factory(taps=BLOCK_TAPS)))
    v_channel_filter: FilterChannel = dataclasses.field(default_factory= lambda : FilterChannel(*butter_filter_factory(taps=BLOCK_TAPS)))
    theta_channel_filter: FilterChannel =  dataclasses.field(default_factory= lambda : FilterChannel(*butter_filter_factory(taps=BLOCK_TAPS)))
    theta_dot_channel_filter: FilterChannel = dataclasses.field(default_factory= lambda : FilterChannel(*butter_filter_factory(taps=BLOCK_TAPS)))


    def __update_block_state(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        pass

    def __output_block_state(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        x_val, v_val, theta_val, theta_dot_val = u
        x_filtered = self.x_channel_filter(t, np.array([x_val]), None, None),
        v_Filtered = self.v_channel_filter(t, np.array([v_val]), None, None),
        theta_filtered = self.theta_channel_filter(t, np.array([theta_val]), None, None)
        theta_dot_filtered = self.theta_dot_channel_filter(t, np.array([theta_dot_val]), None, None)
        return np.array([
            x_filtered[0],
            v_Filtered[0],
            theta_filtered,
            theta_dot_filtered
        ])

    def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, self.__output_block_state,
            name="FilterBlock",
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
    resuit = channel_filter(0, np.array([1]), channel_filter.z_i)
    resuit_2 = channel_filter(0, np.array([3]), channel_filter.z_i)
    print(resuit)
    print(resuit_2)
    filter_block = FilterBlock()
    print(filter_block)
    sys_filter_block = filter_block.as_non_linear_io_system()
    print(sys_filter_block.output(1, np.ones(4), np.ones(4)))
    print(sys_filter_block.output(1, np.ones(4), np.ones(4)))
    print(sys_filter_block.output(1, np.ones(4), np.ones(4)))
    print(sys_filter_block.output(1, np.ones(4), np.ones(4)))