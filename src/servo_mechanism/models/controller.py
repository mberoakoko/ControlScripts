import dataclasses

import control

@dataclasses.dataclass
class PID_Controller:
    k: float
    k_i: float = dataclasses.field(default=0.0)
    k_d: float = dataclasses.field(default=0.0)

    def __as_transfer_function(self):
        s = control.tf([1, 0], [1])
        return self.k + (self.k_i/s) + (s*self.k_d)

    def __as_state_space(self) -> control.StateSpace:
        return control.tf2ss(self.__as_transfer_function(), inputs=["y"], outputs=["tau"])

    def temp_func(self) -> None:
        print(self.__as_transfer_function())
        print(self.__as_state_space())


if __name__ == "__main__":
    pid_controller = PID_Controller(k=10, k_i=0.01, k_d=10)
    pid_controller.temp_func()