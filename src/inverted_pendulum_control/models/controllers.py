import control
import dataclasses
import numpy as np
import abc

@dataclasses.dataclass
class LQRController(abc.ABC):
    plant: control.StateSpace
    Q: np.ndarray = dataclasses.field(default=np.diag([1000, 10, 10000, 10])) # x, v, theta, theta_dot
    R: np.ndarray = dataclasses.field(default=np.eye(5))                   # forcing penalty, _, _, _, _
    K: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.K, _, _ = control.lqr(self.plant.A, self.plant.B, self.Q, self.R)
        eigen_values = np.linalg.eigvals(self.plant.A - self.plant.B @ self.K)
        print("-----" * 30)
        print("Controller Properties")
        print(self.K)
        print(eigen_values)
        print("-----"*30)

    @abc.abstractmethod
    def _output_func(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        raise NotImplementedError("The current function is not implemented")

    @abc.abstractmethod
    def as_non_linear_output_function(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, self._output_func,
            name="LQR_Controller",
            inputs=["x", "v", "theta", "theta_dot"],
            outputs=["u_force"]
        )


@dataclasses.dataclass
class LQR_Stabilizing_Controller(LQRController):
    x_d: np.ndarray = dataclasses.field(default=np.array([0, 0, np.pi, 0]))

    def _output_func(self, t, x: np.ndarray, u: np.ndarray, parmas):
        # print(u)
        # print(f"{u.shape=}")
        # print(f"Controller Output {self.K @ (u - self.x_d)}")
        return -(self.K @ (u - self.x_d))[0]

    def as_non_linear_output_function(self) -> control.NonlinearIOSystem:
        return super().as_non_linear_output_function()


@dataclasses.dataclass
class LQR_CommandFollowind_Controller(abc.ABCMeta, LQRController):
    K2: float = dataclasses.field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.K2 = ... # Formula for calculating the static gain of this signal

    def _output_func(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        ...

    def as_non_linear_output_function(self) -> control.NonlinearIOSystem:
        ...


