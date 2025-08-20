import dataclasses
import numpy as np
import control

@dataclasses.dataclass(frozen=True)
class JetAircraftPlant:
    A: np.ndarray[float] = dataclasses.field(default_factory=lambda :np.array([
        [-0.558, -0.9968, 0.0802, 0.0415],
        [0.5980, -0.1150, -0.0318, 0],
        [-3.05, 0.388, -0.465, 0],
        [0, 0.0805, 1, 0]
    ]))
    B: np.ndarray[float] = dataclasses.field(default_factory=lambda :np.array([
        [0.729, 0.001],
        [-4.75, 1.23],
        [1.53, 10.63],
        [0, 0]
    ]))
    C: np.ndarray[float] = dataclasses.field(default_factory=lambda :np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]))

    def __update(self, t, x: np.ndarray[float], u: np.ndarray[float], params: dict) -> np.ndarray[float]:
        return self.A @ x + self.B @ u

    def __output(self, t, x: np.ndarray[float], u: np.ndarray[float], params: dict) -> np.ndarray[float]:
        return self.C @ x

    def as_nonlinear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__update, self.__output,
            name="JetAircraftPlamt",
            states=["x_1_", "x_2_", "x_3_", "x_4_"],
            inputs=["u_1", "u_2"],
            outputs=["x_1", "x_2", "x_3", "x_4"]
        )