import numpy as np

from models.simple_mass_spring_damper import SimpleMassSpringDamper, create_reference_model

if __name__ == "__main__":
    m_s_p = SimpleMassSpringDamper()
    create_reference_model(m_s_p,q=np.diag([1, 1]), r=np.diag([1]))