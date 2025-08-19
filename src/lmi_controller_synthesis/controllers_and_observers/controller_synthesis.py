import dataclasses

import cvxpy
import cvxpy as cp
from typing import Callable, Any

import control
import numpy as np
from numpy import ndarray


from models.model import JetAircraftPlant

@dataclasses.dataclass()
class SimpleStabilizingController:
    plant: JetAircraftPlamt
    alpha: float = dataclasses.field(default=0.1)
    u: Callable[[np.ndarray], np.ndarray] = dataclasses.field(init=False)

    def __synthesize_controller(self) -> Callable[[np.ndarray], np.ndarray]:
        n = self.plant.A.shape[0]
        m = self.plant.B.shape[1]
        Z = cvxpy.Variable((m, n))
        P = cvxpy.Variable(self.plant.A.shape)
        # zero = cvxpy.Parameter(0)
        objective = cvxpy.Minimize(0)
        constraints = [
            P >> 0,
            self.plant.A @ P + P @ self.plant.A + self.plant.B @ Z + Z.T @ self.plant.B.T + self.alpha * np.eye(n)<< 0
        ]
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(verbose=True)
        gain = Z.value @ np.linalg.inv(P.value)
        print("----"*20, "\n")
        print(f"Closed_loop poles\n{np.linalg.eigvals(self.plant.A + self.plant.B @ gain)}")
        print("----" * 20, "\n")
        def _controller_func(x: np.ndarray) -> np.ndarray:
            return Z.value @ np.linalg.inv(P.value) @ x

        return _controller_func

    def __post_init__(self):
        self.u = self.__synthesize_controller()

    def __controller_output(self, t, x: np.ndarray, u: np.ndarray, params: dict) -> ndarray:
        return self.u(u)

    def as_nonlinear_io_systeem(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, self.__controller_output,
            name="SimpleStabilizingController",
            inputs=["x_1", "x_2", "x_3", "x_4"],
            outputs=["u_1", "u_2"]
        )