import numpy as np
from itertools import product
from typing import Sequence


def gaussian_distribution_initial_states(
    state_shape: Sequence[int], n_states: int, width: float, max_value: float
) -> Sequence[np.ndarray]:
    center = np.array([size // 2 for size in state_shape])

    def generate_points():
        generated = 0
        while generated < n_states:
            point = np.random.rand(len(center)) * 2 - 1
            transformed_point = center + point * center
            if np.linalg.norm(point) <= 1:
                generated += 1
                yield transformed_point.astype(np.int16)

    coordinates = [range(0, size) for size in state_shape]
    states = []

    for point in generate_points():
        state = np.zeros(state_shape)
        for coordinate in product(*coordinates):
            state[coordinate] = np.linalg.norm(point - coordinate)
        state = max_value * np.exp(-1 / width * state**2)
        states.append(state)

    return states
