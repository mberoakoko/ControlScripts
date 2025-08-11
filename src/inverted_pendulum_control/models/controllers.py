import control
import dataclasses
import numpy as np
import abc

@dataclasses.dataclass
class LQRController(abc.ABC):
    plant: control.StateSpace
    Q: np.ndarray = dataclasses.field(default=np.diag([1000, 1000, 1000, 1000])) # x, v, theta, theta_dot
    R: np.ndarray = dataclasses.field(default=0.001*np.eye(5))                   # forcing penalty, _, _, _, _
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
class LQR_CommandFollowind_Controller(LQRController):
    K2: float | np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        super().__post_init__()
        A, B, C = self.plant.A, self.plant.B, self.plant.C
        print(f"{C.shape=}")
        print(B)
        print(B.shape)
        print(f"{(A - B @ self.K).shape=}")
        self.K2 = -C @ np.linalg.inv(A - B @ self.K) @ B # Formula for calculating the static gain of this signal
        self.K2 = np.linalg.pinv(self.K2)

    def test_func(self):
        print(f"{self.K2=}")
        print(f"{self.K2.shape=}")

    def _output_func(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        pend_up_state = np.array([0, 0, np.pi, 0])
        x_state = u[:4]
        c_command = u[4:]
        actuation = -self.K @ (x_state - pend_up_state) + self.K2 @ c_command
        # print(actuation)
        return actuation[0]

    def as_non_linear_output_function(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, self._output_func,
            name="LQR_Command_Following_Controller",
            inputs=["x", "v", "theta", "theta_dot","x_d", "v_d", "theta_d", "theta_dot_d"],
            outputs=["u_force"]
        )


