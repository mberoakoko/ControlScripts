import dataclasses

import cvxpy
import cvxpy as cp
from typing import Callable, Any, NamedTuple

import control
import numpy as np
from numpy import ndarray

from lmi_controller_synthesis.model.models import JetAircraftPlant
# from model.models import JetAircraftPlant

@dataclasses.dataclass()
class SimpleStabilizingController:
    plant: JetAircraftPlant
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

@dataclasses.dataclass
class DSpaceControlLawSynthesizer:
    plant: JetAircraftPlant | control.StateSpace
    rise_time: float
    settling_time: float
    maximum_overshoot: float
    r: float = dataclasses.field(init=False)
    alpha: float = dataclasses.field(init=False)
    c: float = dataclasses.field(init=False)
    A: np.ndarray = dataclasses.field(init=False)
    B: np.ndarray = dataclasses.field(init=False)
    P: cvxpy.Variable = dataclasses.field(init=False)
    Z: cvxpy.Variable = dataclasses.field(init=False)
    K: np.ndarray = dataclasses.field(init=False)



    def __post_init__(self):
        self.r = 1.8**2 / self.rise_time**2
        self.alpha = 4.6/self.settling_time
        self.c = np.pi / np.log(self.maximum_overshoot)
        n = self.plant.A.shape[0]
        m = self.plant.B.shape[1]
        self.Z = cvxpy.Variable((m, n))
        self.P = cvxpy.Variable(self.plant.A.shape, symmetric=True)
        self.A = self.plant.A
        self.B = self.plant.B
        self.K = np.array([0])


    def __rise_time_constraint(self) -> cvxpy.Constraint:
        beta = (self.A@self.P + self.B @ self.Z)
        lmi_mat = cvxpy.bmat([
            [-self.r * self.P, beta],
            [beta.T , -self.r*self.P]
        ])
        return lmi_mat << 0

    def __settling_time_constraint(self) -> cvxpy.Constraint:
        beta = (self.A@self.P + self.B @ self.Z)
        lmi_mat = (beta + beta.T + self.alpha * self.P )
        return  lmi_mat << 0

    def __maximum_overshoot_constraint(self) -> cvxpy.Constraint:
        beta = (self.A @ self.P + self.B @ self.Z)
        lmi_mat = cvxpy.bmat([
            [beta + beta.T, self.alpha * (beta - beta.T)],
            [self.alpha * (beta - beta.T), beta + beta.T]
        ])
        return lmi_mat << 0

    def sysnthesize_constroller(self, espilon: float = 0.001) -> Callable[[np.ndarray], np.ndarray]:
        objective = cvxpy.Minimize(0)
        n = self.plant.A.shape[0]
        constraints = [self.P - espilon * np.eye(n) >> 0 , self.__rise_time_constraint(), self.__settling_time_constraint(), self.__maximum_overshoot_constraint()]
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(verbose=True)
        self.K: np.ndarray = (self.Z @ np.linalg.inv(self.P.value)).value
        def _controller(self, x: np.ndarray) -> np.ndarray:
            return self.K @ x

        return _controller


@dataclasses.dataclass
class TrajectoryFollowingController:
    sythesizer: DSpaceControlLawSynthesizer
    control_law: Callable[[np.ndarray], np.ndarray] = dataclasses.field(init=False)
    k_2: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.control_law = self.sythesizer.sysnthesize_constroller()
        self.k_2 = self.__calculate_feed_forward_gain()

    def __calculate_feed_forward_gain(self) -> np.ndarray:
        plant = self.sythesizer.plant
        k_2: np.ndarray = -np.linalg.pinv(plant.C  @ np.linalg.inv(plant.A + plant.B @ self.sythesizer.K))
        print(f"{k_2.shape=}")
        return k_2


    def __controller_output(self, t, x: np.ndarray, u: np.ndarray, params: dict) -> ndarray:
        x_state = u[:4]
        command = u[4:]
        return self.control_law(x_state) + self.k_2 @ command
        ...

    def as_nonlinear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, self.__controller_output,
            name="TrajectoryFollowingController",
            inputs=["x_1", "x_2", "x_3", "x_4", "c_1", "c_2"],
            outputs=["u_1", "u_2"]
        )


class NineMatrixData(NamedTuple):
    A: np.ndarray
    B_1: np.ndarray
    B_2: np.ndarray
    C_1: np.ndarray
    C_2: np.ndarray
    D_1_1: np.ndarray
    D_1_2: np.ndarray
    D_2_1: np.ndarray
    D_2_2: np.ndarray

@dataclasses.dataclass
class FullStateOptimalController:
    params: NineMatrixData
    Y: cvxpy.Variable = dataclasses.field(init=False)
    Z: cvxpy.Variable = dataclasses.field(init=False)

    def __post_init__(self):
        self.Y = cvxpy.Variable(self.params.A.shape)
        self.Z = cvxpy.Variable((self.params.B_2.shape[1], self.params.A.shape[0]))

    def __sythesize_controller(self) -> np.ndarray:
        gamma = cvxpy.Variable()

        alpha = self.Y @ self.params.A.T + self.Z.T @ self.params.B_2.T + self.params.B_2@self.Z
        beta = self.Y @ self.params.C_1 + self.Z.T @ self.params.D_1_2.T
        eye_1 = np.eye(self.params.B_1.T.shape[0], self.params.B_1.shape[1])
        eye_2 = np.eye(self.params.D_1_1.shape[0], self.params.D_1_1.T.shape[1])
        row_1 = [-self.Y , np.zeros((self.Y.shape[0], self.params.B_1.T.shape[1])), np.zeros((self.Y.shape[0], self.params.B_1.shape[1])), np.zeros((self.Y.shape[0], self.params.D_1_1.T.shape[1]))]
        row_2 = [np.zeros(self.Y.shape),alpha, self.params.B_1, beta ]
        row_3 = [np.zeros((self.params.B_1.T.shape[0], self.Y.shape[1])), self.params.B_1.T, -gamma*eye_1, self.params.D_1_1.T]
        row_4 = [np.zeros((self.params.D_1_1.shape[0], self.Y.shape[1])), beta.T, self.params.D_1_1, -gamma*eye_2]
        constraints = [cvxpy.bmat([row_1, row_2, row_3, row_4]) << 0]
        problem = cvxpy.Problem(cvxpy.Minimize(gamma), constraints=constraints)
        problem.solve(verbose=True)
        return self.Z.value @ np.linalg.inv(self.Y.value)

    def f_matrix(self):
        return self.__sythesize_controller()