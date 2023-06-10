import numpy as np
from typing import Sequence

from physicslearn.equations.abstract import PhyDiffEq


def generate_data(
    eq: PhyDiffEq,
    eq_parameters: np.ndarray,
    time_range: tuple[int, int, int],
    initial_states: Sequence[np.ndarray],
) -> np.ndarray:
    # TODO - support for PhyDiffEq parameter
    states = np.array(initial_states)
    T = np.ones((len(initial_states),))
    data = []
    for t in range(time_range[0], time_range[1]):
        if not t % time_range[2]:
            data.append(states.copy())
            T.fill(t)
            states += eq(states, T, eq_parameters)
    return np.swapaxes(np.array(data), 0, 1)
