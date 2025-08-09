import control
import numpy as np
import dataclasses
import enum

class MeasurementType(enum.Enum):
    FULLSTATE: int = 1
    PARTIAL_STATE: int = 0

@dataclasses.dataclass
class InvertedPendulum:
    m: float = dataclasses.field(default=10)
    M: float = dataclasses.field(default=2)
    l: float = dataclasses.field(default=2)
    g: float = dataclasses.field(default=9.8)
    d: float = dataclasses.field(default=0.2)

    def __pendulum_state_update(self, t: float, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        x, v, theta, omega = x
        u_force = u[0]
        s_x = np.sin(theta)
        c_x = np.cos(theta)
        d = self.m * self.l*self.l * (self.m + self.M * (1 - c_x**2))
        beta = (self.m*self.l*omega**2*s_x-self.d*v)
        dx = np.array([
            v,
            (-(self.m**2 * self.l**2*self.g*s_x*c_x)+self.m*self.l*beta+(self.M*self.l**2*u_force))/d,
            omega,
            ((self.m + self.M)*(self.m*self.g*self.l*s_x) - self.m*self.l*c_x*beta-self.m*self.l*c_x*u_force)/d
        ])
        return dx


    def __pendulum_state_output(self, t: float, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        w_n = u[1:]
        return np.array([x + w_n])

    def __partial_state_output(self, t: float, x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
        w_n = u[1:]
        x_new = x + w_n
        return np.array([x_new[0], x_new[2]])

    def as_non_linear_io_system_full_state_measurement(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__pendulum_state_update, self.__pendulum_state_output,
            name="PendulumPlant",
            states=["x", "v", "theta", "theta_dot"],
            inputs=["u_force", "x_n", "v_n", "theta_n", "theta_dot_n"],
            outputs=["x", "v", "theta", "theta_dot"]
        )

    def as_non_linear_io_system_partial_state_measurement(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__pendulum_state_update, self.__partial_state_output,
            name="PendulumPlant",
            states=["x", "v", "theta", "theta_dot"],
            inputs=["u_force", "x_n", "v_n", "theta_n", "theta_dot_n"],
            outputs=["x", "theta"]
        )

def linearize_plant(inverted_pendulum_system: InvertedPendulum, measurement: MeasurementType = MeasurementType.FULLSTATE)-> control.StateSpace:
    """
    Function linearizes plant depending on what type of measurements we are getting from the plant
    :param inverted_pendulum_system: Inverted Pendulum Struct
    :param measurement: measurement type
    :return: inverted pendulum as a state space system
    """
    def plant_factory(measurement_:MeasurementType) ->control.NonlinearIOSystem:
        inner_plant = None
        match measurement_:
            case MeasurementType.FULLSTATE:
                inner_plant =  inverted_pendulum_system.as_non_linear_io_system_full_state_measurement()
            case MeasurementType.PARTIAL_STATE:
                inner_plant = inverted_pendulum_system.as_non_linear_io_system_partial_state_measurement()
        return inner_plant

    x_eq: np.ndarray = np.array([0, 0, np.pi, 0])
    u_eq: np.ndarray = np.zeros(5)
    return control.linearize(plant_factory(measurement), x_eq.tolist(), u_eq.tolist())

if __name__ == "__main__":
    test_inverted_pendulum = InvertedPendulum()
    print(test_inverted_pendulum.as_non_linear_io_system_full_state_measurement())
    print(linearize_plant(test_inverted_pendulum))
    plant = linearize_plant(test_inverted_pendulum, measurement=MeasurementType.PARTIAL_STATE)
    print(f"Controllability_rank {np.linalg.matrix_rank(control.ctrb(plant.A, plant.B))}")
    print(f"Observerbility_rank {np.linalg.matrix_rank(control.obsv(plant.A, plant.C))}")

