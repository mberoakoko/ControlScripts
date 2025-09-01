import control
from typing import NamedTuple

import cvxpy
import numpy as np
import cvxpy as cp

class GenericSystem(NamedTuple):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

def estimate_h_infinity(system: GenericSystem):
    gamma: cp.Variable = cp.Variable(shape=(1, 1))
    X: cp.Variable = cp.Variable(shape=system.A.shape)
    identity: np.ndarray = np.eye(system.A.shape[0])
    objective = cvxpy.Minimize(0)
    block_a = cp.bmat([
        [system.A.T @ X + system.A @ X, X @ system.B],
        [system.B.T @ X, -gamma * identity],
    ])
    block_b = cp.bmat([
        [system.C.T],
        [system.D.T]
    ])
    block_c = cp.bmat([[system.C, system.D]])
    constraints = [
        X >> 0,
        block_a + (1/gamma) * block_b.T @ block_c << 0,
    ]

    problem = cvxpy.Problem(objective, constraints)
    problem.solve(verbose=True)
    return gamma.value, X.value