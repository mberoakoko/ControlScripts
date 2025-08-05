import control
import dataclasses

import numpy as np


@dataclasses.dataclass
class LQRController:
    linearized_plant: control.StateSpace
    Q: np.ndarray = dataclasses.field(default=np.diag([1, 1, 1])) # X Y Theta Weighting
    R: np.ndarray = dataclasses.field(default=np.diag([0.001, 1])) # velocity delta
    K: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.K, _, _ = control.lqr(self.linearized_plant, self.Q, self.R)
        closed_loop_dynamics = self.linearized_plant.A - self.linearized_plant.B @ self.K
        print(f"Eigen Values {np.linalg.eigvals(closed_loop_dynamics)}")

    def __controller_output(self, t, x: np.ndarray, z: np.ndarray, params) -> np.ndarray:
        x_d_vec = z[:3]
        u_d_vec = z[3:5]
        x_vec = z[5:]
        return u_d_vec - self.K @ (x_d_vec - x_vec)

    def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, self.__controller_output, name="LQR_controller",
            inputs=("x_d","y_d", "theta_d", "v_d", "delta_d", "x", "y", "theta"),
            outputs=("v", "delta")
        )

