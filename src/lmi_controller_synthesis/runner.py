from model.models import JetAircraftPlamt
from controllers_and_observers.controller_synthesis import SimpleStabilizingController

import control
import numpy as np

if __name__ == "__main__":
    jet_plant_raw = JetAircraftPlamt()
    jet_plant = jet_plant_raw.as_nonlinear_io_system()
    print(jet_plant)
    print(np.linalg.matrix_rank(control.ctrb(jet_plant_raw.A, jet_plant_raw.B)))
    print(np.linalg.matrix_rank(control.obsv(jet_plant_raw.A, jet_plant_raw.C)))

    simple_controller_raw = SimpleStabilizingController(
        plant=jet_plant_raw,
    )

    simple_controller = simple_controller_raw.as_nonlinear_io_systeem()
    print(simple_controller)