import control

from ..model.models import JetAircraftPlant
from ..controllers_and_observers.controller_synthesis import SimpleStabilizingController

def create_simple_stable_closed_loop_plant() -> control.NonlinearIOSystem:
    jet_aircraft_plant_raw = JetAircraftPlant()
    jet_aircraft_controller = SimpleStabilizingController(
        plant=jet_aircraft_plant_raw,
    )
    return control.interconnect(
        syslist=[jet_aircraft_plant_raw.as_nonlinear_io_system(), jet_aircraft_controller.as_nonlinear_io_systeem()],
        inputs=[""],
        outputs=[""]
    )