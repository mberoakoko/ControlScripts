import dataclasses
import numpy as np
import control

from src.adaptive_controllers.models.simple_mass_spring_damper import MassSpringDamperLinearParams
from steering_control.models.system_dynamics import DelayBlock


@dataclasses.dataclass(frozen=True)
class JetAircraftPlant:
    A: np.ndarray[float] = dataclasses.field(default_factory=lambda :np.array([
        [-0.558, -0.9968, 0.0802, 0.0415],
        [0.5980, -0.1150, -0.0318, 0],
        [-3.05, 0.388, -0.465, 0],
        [0, 0.0805, 1, 0]
    ]))
    B: np.ndarray[float] = dataclasses.field(default_factory=lambda :np.array([
        [0.729, 0.001],
        [-4.75, 1.23],
        [1.53, 10.63],
        [0, 0]
    ]))
    C: np.ndarray[float] = dataclasses.field(default_factory=lambda :np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]))

    def __update(self, t, x: np.ndarray[float], u: np.ndarray[float], params: dict) -> np.ndarray[float]:
        return self.A @ x + self.B @ u

    def __output(self, t, x: np.ndarray[float], u: np.ndarray[float], params: dict) -> np.ndarray[float]:
        return self.C @ x

    def as_nonlinear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__update, self.__output,
            name="JetAircraftPlamt",
            states=["x_1_", "x_2_", "x_3_", "x_4_"],
            inputs=["u_1", "u_2"],
            outputs=["x_1", "x_2", "x_3", "x_4"]
        )

@dataclasses.dataclass
class MassSpringDamperExogenous:
    m: float
    c: float
    k: float
    alpha_1: float
    alpha_2: float
    A: np.ndarray = dataclasses.field(init=False)
    B_1: np.ndarray = dataclasses.field(init=False)
    B_2: np.ndarray = dataclasses.field(init=False)
    C_1: np.ndarray = dataclasses.field(init=False)
    C_2: np.ndarray = dataclasses.field(init=False)
    D_1_1: np.ndarray = dataclasses.field(init=False)
    D_1_2: np.ndarray = dataclasses.field(init=False)
    D_2_1: np.ndarray = dataclasses.field(init=False)
    D_2_2: np.ndarray = dataclasses.field(init=False)
    p: float = dataclasses.field(init=False)
    q: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.A = np.array([
            [0, 1],
            [-self.k/self.m, -self.c/self.m],
        ])

        self.B_1 = np.array([
            [0],
            [1/self.m],
        ])

        self.B_2 = np.array([
            [0],
            [1/self.m],
        ])

        self.C_1 = np.array([
            [self.alpha_1, 0],
            [0, 0]
        ])
        self.C_2 = np.array([
            [1, 0]
        ])
        self.D_1_1 = np.array([
            [0],
            [0]
        ])
        self.D_1_2 = np.array([
            [0],
            [self.alpha_2]
        ])

        self.D_2_1 = np.array([
            [0]
        ])

        self.D_2_2 = np.array([
            [0]
        ])

        self.p = self.B_1.shape[1]
        self.q = self.A.shape[0]


    def __plant_update(self,t, x: np.ndarray[float], u: np.ndarray[float], params: dict) -> np.ndarray:
        u_forcing = np.expand_dims(u[:self.p], axis=0)
        w_forcing = np.expand_dims(u[self.p:], axis=0)
        return self.A @ x + np.squeeze(self.B_1 @ u_forcing) + np.squeeze(self.B_2 @ w_forcing)

    def __plant_output(self, t, x: np.ndarray[float], u: np.ndarray[float], params: dict) -> np.ndarray:
        u_forcing = u[:self.p]
        w_forcing = u[self.p:]
        result = np.r_[
            self.C_1 @ x + self.D_1_1 @ u_forcing + self.D_1_2 @ w_forcing,
            self.C_2 @ x + self.D_2_1 @ u_forcing + self.D_2_2 @ w_forcing
        ]
        return result

    def as_non_linear_io_system(self) -> control.NonlinearIOSystem:
         return control.NonlinearIOSystem(
             self.__plant_update, self.__plant_output,
             name="MassSpringDamperExogenousForcing",
             states=["x", "x_dot"],
             inputs=["w", "u"],
             outputs=["z", "y_1_prime", "y_2_prime"],
         )

@dataclasses.dataclass
class DelayBlock:
    time_delay: float = dataclasses.field(default=0.001)
    order: int  = dataclasses.field(default=3)
    channel_1: control.StateSpace = dataclasses.field(init=False)
    channel_2: control.StateSpace = dataclasses.field(init=False)
    delay_block: control.StateSpace = dataclasses.field(init=False)


    def __post_init__(self):
        self.channel_1 = control.TransferFunction(*control.pade(self.time_delay, self.order)).to_ss()
        self.channel_2 = control.TransferFunction(*control.pade(self.time_delay, self.order)).to_ss()
        self.delay_block = control.append(self.channel_1, self.channel_2)

    def as_nonlinear_io_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.delay_block.dynamics, self.delay_block.output,
            name="DelayBlock",
            states=self.delay_block.nstates,
            inputs=["y_1_prime", "y_2_prime"],
            outputs=["y_1", "y_2"],
            params=None
        )

if __name__ == "__main__":
    m_s_p_system = MassSpringDamperExogenous(
        m=1,
        c=1,
        k=1,
        alpha_1=1,
        alpha_2=1,
    )
    print(m_s_p_system.as_non_linear_io_system())
    sys_dyn = m_s_p_system.as_non_linear_io_system()
    excitation_signal = np.array([0, 0])
    print(excitation_signal)
    print(f"{sys_dyn.dynamics(0, np.array([0, 0]), excitation_signal)=}")
    print(f"{sys_dyn.output(0, np.array([0, 0]), excitation_signal)=}")
    delay_block = DelayBlock(
        time_delay=0.01
    ).as_nonlinear_io_system()
    sys_with_delay = control.interconnect([sys_dyn, delay_block])
    print(sys_with_delay)