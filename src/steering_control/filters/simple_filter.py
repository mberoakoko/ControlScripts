import control
import numpy as np
import dataclasses

def create_dummy_system()-> control.StateSpace:
    A = np.array([[0, 1], [-10, -2]])
    B = np.array([[0], [10]])
    C = np.array([[1, 0]])
    D = np.array([[0]])
    return control.ss(A, B, C, D)

def low_pass_statespace(tau: float = 0.1)->control.StateSpace:
    A_f = np.array([[-1 / tau]])
    B_f = np.array([[1 / tau]])
    C_f = np.array([[1]])
    D_f = np.array([[0]])
    return control.ss(A_f, B_f, C_f, D_f)

@dataclasses.dataclass
class AugmentedSystem:
    A_aug: np.ndarray = dataclasses.field(init=False)
    B_aug: np.ndarray = dataclasses.field(init=False)
    C_aug: np.ndarray = dataclasses.field(init=False)
    D_aug: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        plant = create_dummy_system()
        A, B, C, D = plant.A, plant.B, plant.C, plant.D
        filtering_sys = low_pass_statespace()
        A_f, B_f, C_f = filtering_sys.A, filtering_sys.B, filtering_sys.C
        self.A_aug = np.block([
            [A, np.zeros((A.shape[0], A_f.shape[0]))],
            [np.dot(B_f, C), A_f]
        ])
        self.B_aug = np.block([
            [B],
            [np.zeros((B_f.shape[0], B.shape[1]))]
        ])
        self.C_aug = np.block([
            [np.zeros((C_f.shape[0], A.shape[1])), C_f]
        ])
        self.D_aug = np.zeros((C_f.shape[0], B.shape[1]))

    def __update_function(self, t, x: np.ndarray, u: np.ndarray, params)-> np.ndarray:
        return self.A_aug @ x + self.B_aug @ u

    def __output_function(self, t, x:np.ndarray, u: np.ndarray, params) -> np.ndarray:
        return self.C_aug @ x + self.D_aug @ u

    def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__update_function, self.__output_function,
            ...
        )


def pliriminary_checks() -> None:
    aug_sys = AugmentedSystem()
    print(aug_sys)



if __name__ == "__main__":
    pliriminary_checks()