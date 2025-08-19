from model.models import JetAircraftPlamt
import control
import numpy as np

if __name__ == "__main__":
    jet_plant_raw = JetAircraftPlamt()
    jet_plant = jet_plant_raw.as_nonlinear_io_system()
    print(jet_plant)
    print(np.linalg.matrix_rank(control.ctrb(jet_plant_raw.A, jet_plant_raw.B)))
    print(np.linalg.matrix_rank(control.obsv(jet_plant_raw.A, jet_plant_raw.C)))