from model.models import JetAircraftPlant
from controllers_and_observers.controller_synthesis import SimpleStabilizingController, DSpaceControlLawSynthesizer, TrajectoryFollowingController

import control
import numpy as np

if __name__ == "__main__":
    jet_plant_raw = JetAircraftPlant()
    jet_plant = jet_plant_raw.as_nonlinear_io_system()
    print(jet_plant)
    print(np.linalg.matrix_rank(control.ctrb(jet_plant_raw.A, jet_plant_raw.B)))
    print(np.linalg.matrix_rank(control.obsv(jet_plant_raw.A, jet_plant_raw.C)))

    simple_controller_raw = SimpleStabilizingController(
        plant=jet_plant_raw,
    )

    simple_controller = simple_controller_raw.as_nonlinear_io_systeem()
    print(simple_controller)
    print("\n\n")
    print("----"*20)
    print("\n\n")
    d_space_constraints = DSpaceControlLawSynthesizer(
        plant=jet_plant_raw,
        rise_time=0.01,
        settling_time=2,
        maximum_overshoot=0.1
    )

    print(d_space_constraints)
    print(d_space_constraints.sysnthesize_constroller())

    traj_controller = TrajectoryFollowingController(
        sythesizer=d_space_constraints
    )

    print("====="*20)