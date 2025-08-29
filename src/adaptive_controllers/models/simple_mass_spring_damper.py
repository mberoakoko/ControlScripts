from email.policy import default
from sys import deactivate_stack_trampoline

import control
import dataclasses

import numpy as np


@dataclasses.dataclass
class SimpleMassSpringDamper:
    m: float = dataclasses.field(default=5.0)
    alpha: float = dataclasses.field(default=0.1)
    beta: float = dataclasses.field(default=0.1)

    def __plant_update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        x_1, x_2 = x
        return np.array([
            x_2,
            (1/self.m) * (u - self.alpha * x_1 - self.beta * x_2**2)
        ])

    def __plant_output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return x

    def get_linear_params(self) -> MassSpringDamperLinearParams:
        return MassSpringDamperLinearParams(
            A=np.array([
                [0, 1],
                [0, 0]
            ]),
            B=np.array([[0], [1]]),
            Nabla=np.array([[1/self.m]]),
        )

    def as_non_linear_plant(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__plant_update, self.__plant_output,
            name="SimpleMassSpringDamper",
            inputs=["u"],
            outputs=["x", "x_dot"]
        )


def create_reference_model(m_s_p: SimpleMassSpringDamper, q: np.ndarray , r: np.ndarray) -> control.StateSpace:
    sys_parms = m_s_p.get_linear_params()
    k, _, eig = control.lqr(sys_parms.A, sys_parms.B , q, r)
    print(eig)
    c = np.eye(sys_parms.A.shape[0])
    k_2: np.ndarray = np.linalg.pinv(-c @ np.linalg.inv(sys_parms.A - sys_parms.B @ k) @ sys_parms.B)
    return control.StateSpace(sys_parms.A - sys_parms.B @ k , sys_parms.B @ k_2 , c , np.zeros_like(sys_parms.B))