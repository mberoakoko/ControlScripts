from models.servo_mechanism_model import ServoMechanismModel
from plotting_ops import plot_mech_system_response
from simulator import simulate_impulse_response_model

def run_impulse_respose_simulation() -> None:
    plot_mech_system_response(
        respose_data=simulate_impulse_response_model(
            dt=0.01,
            t_final=5,
            model=ServoMechanismModel()
        )
    )

if __name__ == "__main__":
    run_impulse_respose_simulation()