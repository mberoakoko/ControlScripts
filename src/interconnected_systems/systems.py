import dataclasses
import control
import numpy as np
from typing import TypeAlias

SIMULATION_DT = 0.02

@dataclasses.dataclass
class PlantFactory:
    dt: float

    @staticmethod
    def create() -> control.TransferFunction:
        return control.tf((0.5, ), (0.1, 1), inputs="u", outputs="y")

    def discretize(self, dt: float | None = None) -> control.TransferFunction:
        dt = dt if dt  else self.dt
        return control.c2d(PlantFactory.create(), dt, "zoh")

ControllerSystemType: TypeAlias = control.StateSpace | control.TransferFunction

class SampledDataController:
    class Details:
        def __init__(self, controller: control.StateSpace | control.TransferFunction, plant_dt: float):
            self.__x: np.ndarray = np.array([])
            self.__y: np.ndarray = np.zeros((controller.noutputs, 1))
            self.__step: int = 0
            self.__n_steps: int = int(round(controller.dt/plant_dt))
            self.__controller: ControllerSystemType = controller

        def update_function(self, t, x, u, params):
            if self.__step == 0:
                self.__x: np.ndarray = self.__controller.dynamics(t, x, u, params)
            self.__step += 1
            if self.__step == self.__n_steps:
                self.__step = 0
            return self.__x.copy()

        def output_function(self, t, x, u, params):
            if self.__step == 0:
                self.__y: np.ndarray = self.__controller.output(t, x, u, params)
            return self.__y.copy()


    def __init__(self, controller: ControllerSystemType, plant_dt: float):
        self.__plant_dt = plant_dt
        assert control.isdtime(controller, True), "Controller Must be in discrete time"
        self.__controller: control.StateSpace = control.ss(controller)
        one_plus_eps = 1 + np.finfo(float).eps
        assert np.isclose(0, self.__controller.dt * one_plus_eps % plant_dt), \
            "plant_dt must be an integral multiple of the controller's dt"
        self.__sampler = SampledDataController.Details(self.__controller, plant_dt)

    def create(self) -> control.InputOutputSystem:
        return control.ss(
            self.__sampler.update_function, self.__sampler.output_function,
            dt=self.__plant_dt, name=self.__controller.name, inputs=self.__controller.input_labels,
            outputs=self.__controller.output_labels, states=self.__controller.state_labels
        )

def create_closed_loop_system() -> control.InputOutputSystem:
    controller_ts = 0.2
    controller = control.tf(1, [1, -.9], controller_ts, inputs='e', outputs='u')
    controller_sim = SampledDataController(controller, SIMULATION_DT).create()
    plant_continuous = PlantFactory.create()
    u_summer = control.summing_junction(inputs=["-y", "r"], output="e")
    plant_simulator = control.c2d(plant_continuous, SIMULATION_DT, "zoh")
    closed_loop_simulator = control.interconnect(
        [controller_sim, plant_simulator, u_summer],
        inputs="r", outputs=["y", "u"])
    return closed_loop_simulator


@dataclasses.dataclass
class DelaySystem:
    delay: float
    dt: float
    inputs: list[str]
    outputs: list[str]

    def generate_system(self) -> control.InputOutputSystem:
        assert self.delay >= 0 , "System must have a positive delay"
        n = int(round(self.delay/self.dt))
        n_inputs = len(self.inputs)
        assert n_inputs == 1, "We only support one input"
        A = np.eye(n, k=-1)
        B = np.eye(n, 1)
        C = np.eye(1, n, k=n-1)
        D = np.zeros((1, 1))
        return control.ss(A, B, C, D, self.dt, inputs=self.inputs, outputs=self.outputs)




if __name__ == "__main__":
    print(create_closed_loop_system())
    ...