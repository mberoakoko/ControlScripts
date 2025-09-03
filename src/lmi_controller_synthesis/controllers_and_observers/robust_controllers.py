import control
import dataclasses
import enum
import numpy as np



@dataclasses.dataclass
class LowerStarController:
    params: control.StateSpace

    def __controller_update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return self.params.A @ x + self.params.B @ u

    def __controller_output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        print(f"{x=}")
        print(f"{x.shape=}")
        return self.params.C @ x + self.params.D @ u

    def as_non_linear_io_system(self, contoller_inputs: list[str] = ("y_1", "y_2" ), controller_outputs: list[str] = ("u", )) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__controller_update, self.__controller_output,
            name="LowerStarController",
            states=["x_ctrl"],
            inputs=contoller_inputs,
            outputs=controller_outputs
        )

