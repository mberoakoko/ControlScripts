import control
import dataclasses
import enum
import numpy as np

class ControllerType(enum.IntEnum):
    FULL_STATE_CONTROLLER = 0
    DYNAMIC_UPDATE_CONTROLLER = 1

@dataclasses.dataclass
class LowerStarController:
    params: control.StateSpace

    def __controller_update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return self.params.A @ x + self.params.B @ u

    def __controller_full_state_output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return self.params.D @ u

    def __controller_dynamic_output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return self.params.C @ x + self.params.D @ u

    def __controller_factory(self,
                             controller_type: ControllerType,
                             controller_inputs: list[str],
                             controller_outputs: list[str],
                             ) -> control.NonlinearIOSystem:
        match controller_type:
            case ControllerType.FULL_STATE_CONTROLLER:
                return control.NonlinearIOSystem(
                    None, self.__controller_full_state_output,
                    name="LowerStarControllerFullState",
                    inputs=controller_inputs,
                    outputs=controller_outputs
                )

            case ControllerType.DYNAMIC_UPDATE_CONTROLLER:
                return control.NonlinearIOSystem(
                    self.__controller_update,self.__controller_dynamic_output,
                    name="LowerStarControllerDynamicUpdate",
                    states=["x_ctrl"],
                    inputs=controller_inputs,
                    outputs=controller_outputs
                )

    def as_non_linear_io_system(self,
                                contoller_inputs: list[str] = ("y_1", "y_2" ),
                                controller_outputs: list[str] = ("u_prime", ),
                                controller_type: ControllerType = ControllerType.FULL_STATE_CONTROLLER) -> control.NonlinearIOSystem:
        return self.__controller_factory(controller_type, contoller_inputs, controller_outputs, )

