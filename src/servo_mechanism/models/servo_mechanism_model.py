import control
import dataclasses

import numpy as np


@dataclasses.dataclass
class ServoMechanismModel:
    J: float    = dataclasses.field(default=100)    # Moment of inertia
    b: float    = dataclasses.field(default=10)     # Angular damping term
    k: float    = dataclasses.field(default=1)      # Spring constant
    r: float    = dataclasses.field(default=1)      # Location of spring constant arm
    l: float    = dataclasses.field(default=2)      # Distance to read head
    eps: float  = dataclasses.field(default=0.1)    # Magnitude of velocity dependent permutation


    def __servo_mech_state_update(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        theta, theta_dot = x
        tau = u[0]
        d_thetadot = 1 / self.J * (-self.b * theta_dot - self.k * self.r * np.sin(theta) + tau)
        return d_thetadot

    def __servo_mech_output(self, t, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        theta, theta_dot = u
        return np.array([self.l * theta - self.eps * theta_dot])

    def __servo_mech_full_state(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__servo_mech_state_update, None,
            name="ServoMechFullState",
            states=["theta_", "theta_dot_"],
            inputs=["tau"],
            outputs=["theta", "theta_dot"]
        )

    def __servo_mech_sensor(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, self.__servo_mech_output,
            name="ServoMechSensor",
            inputs=["theta", "theta_dot"],
            outputs=["y"]
        )

    def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
        return control.interconnect(
            [self.__servo_mech_full_state(), self.__servo_mech_sensor()],
            name="ServoMechanism",
            inputs=["tau"],
            outputs=["y"],
            states=["theta", "theta_dot"],
        )


def linearize_servo_mechanism(servo_mechanism_model: ServoMechanismModel, theta_eq: float = 15) -> control.StateSpace:
    return control.linearize(servo_mechanism_model.as_non_linear_io_system(), [0, 0], [np.deg2rad(15)])

if __name__ == "__main__":
    servo_mechanism = ServoMechanismModel()
    print(servo_mechanism.as_non_linear_io_system())
    linearized_system = linearize_servo_mechanism(servo_mechanism)
    print(linearized_system)
    print(np.linalg.eigvals(linearized_system.A))
