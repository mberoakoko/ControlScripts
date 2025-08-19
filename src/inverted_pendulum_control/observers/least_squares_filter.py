import control
import dataclasses

import numpy as np


@dataclasses.dataclass
class LeastSquaresFilter:
    plant_discr: control.StateSpace
    P: np.ndarray = dataclasses.field(default_factory=lambda :2 * np.diag([1, 13, 1, 1]))
    R: np.ndarray = dataclasses.field(default_factory=lambda :np.diag([10, 12, 13, 7]))
    with_tracing: bool = dataclasses.field(default=False)
    K: np.ndarray = dataclasses.field(init=False)
    P_trace: np.ndarray = dataclasses.field(init=False, repr=False)
    K_trace: np.ndarray = dataclasses.field(init=False, repr=False)

    def __gain_update_law(self) -> np.ndarray:
        alpha = self.plant_discr.C @ self.P @ self.plant_discr.C.T + self.R
        return self.P @ self.plant_discr.C.T @ np.linalg.inv(alpha)


    def __post_init__(self):
        self.K = self.__gain_update_law()
        if self.with_tracing:
            self.P_trace = np.array([self.P.copy().reshape(-1)])
            self.K_trace = np.array([self.K.copy().reshape(-1)])
            np.append(self.K_trace, [self.K.copy().reshape(-1)], axis=0)

    def __covariance_update_law(self) -> np.ndarray:
        beta = np.eye(self.plant_discr.A.shape[0]) - self.K @ self.plant_discr.C
        return beta @ self.P @ beta.T + self.K @ self.R @ self.K.T

    def __least_squares_update(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        self.K = self.__gain_update_law()
        if self.with_tracing:
            np.append(self.K_trace, self.K.copy().reshape(-1))
        return x + self.K @ ( u - self.plant_discr.C @ x )

    def __least_square_output(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        self.P = self.__covariance_update_law()
        if self.with_tracing:
            np.append(self.P_trace, self.P_trace.copy().reshape(-1))
        return x

    def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__least_squares_update, self.__least_square_output,
            name="LeastSquaresFilter",
            states=["x_hat", "v_hat", "theta_hat", "theta_dot_hat"],
            inputs=["x", "v", "theta", "theta_dot"],
            outputs=["x_hat", "v_hat", "theta_hat", "theta_dot_hat"]
        )
