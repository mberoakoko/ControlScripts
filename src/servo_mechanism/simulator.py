import control
import numpy as np

from models.servo_mechanism_model import ServoMechanismModel

def simulate_impulse_response_model(dt: float, t_final: float, model: ServoMechanismModel) -> control.TimeResponseData:
    t_sim = np.linspace(0, t_final, round(t_final/dt))
    forcing = np.ones_like(t_sim)
    return control.input_output_response(model.as_non_linear_io_system(), t_sim, forcing)

if __name__ == "__main__":
    respnse_data = simulate_impulse_response_model(
        dt=0.01,
        t_final=1,
        model=ServoMechanismModel()
    )
    print(respnse_data.to_pandas().to_string())