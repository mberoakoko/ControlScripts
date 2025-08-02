import dataclasses
import control
import numpy as np

SIMULATION_DT = 0.02

@dataclasses.dataclass
class PlantFactory:
    dt: float

    @staticmethod
    def create() -> control.TransferFunction:
        return control.tf((1, ), (0.03, 1), inputs="u", outputs="y")

    def discretize(self, dt: float | None = None) -> control.TransferFunction:
        dt = dt if dt  else self.dt
        return control.c2d(PlantFactory.create(), dt, "zoh")


class SampledDataController:
    class Details:
        def __init__(self, controller: control.StateSpace | control.TransferFunction, plant_dt: float):
            self.__x: np.ndarray = np.array([])
            self.__y: np.ndarray = np.zeros((controller.noutputs, 1))
            self.__step: int = 0
            self.__n_steps: int = int(round(controller.dt/plant_dt))
            self.__controller: control.NonlinearIOSystem = controller

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


    def __init__(self, controller: control.NonlinearIOSystem, plant_dt: float):
        self.__plant_dt = plant_dt
        assert control.isdtime(controller, True), "Controller Must be in discrete time"
        self.__controller: control.StateSpace = control.ss(controller)
        one_plus_eps = 1 + np.finfo(float).eps
        assert np.isclose(0, self.__controller.dt * one_plus_eps % plant_dt), \
            "plant_dt must be an integral multiple of the controller's dt"
        self.__sampler = SampledDataController.Details(self.__controller, plant_dt)

    def create(self) -> control.StateSpace:
        return control.ss(
            self.__sampler.update_function, self.__sampler.output_function,
            dt=self.__plant_dt, name=self.__controller.name, inputs=self.__controller.input_labels,
            outputs=self.__controller.output_labels, states=self.__controller.state_labels
        )


def try_sampled_controller():
    controller_ts = 0.1
    controller = control.tf(1, [1, -.9], controller_ts, inputs='e', outputs='u')
    controller_sim = SampledDataController(controller, SIMULATION_DT)
    print(controller_sim.create())


if __name__ == "__main__":
    try_sampled_controller()