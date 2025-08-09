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