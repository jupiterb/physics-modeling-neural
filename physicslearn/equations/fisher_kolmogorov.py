import numpy as np
from typing import Optional, Sequence

from physicslearn.equations.abstract import PhyDiffEq


class FisherKolmogorovEq(PhyDiffEq):
    def __init__(self, D: float = 0.5, k: float = 0.5) -> None:
        self._D = D
        self._k = k

    def _du_dt(
        self, U: np.ndarray, T: np.ndarray, parameters: Optional[np.ndarray]
    ) -> np.ndarray:
        D, k = self._get_D_k(U.shape, parameters)

        du_dxs = [
            np.gradient(np.gradient(U, axis=x), axis=x) for x in range(1, len(U.shape))
        ]
        return D * sum(du_dxs) + k * U * (1 - U)

    def _get_D_k(
        self, U_shape: Sequence[int], parameters: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = (U_shape[0], *[1 for _ in U_shape[1:]])
        if parameters is None:
            # use default parameters
            return np.ones(shape) * self._D, np.ones(shape) * self._k
        else:
            # use provided parameters
            return (
                parameters[:, 0].reshape(shape),
                parameters[:, 1].reshape(shape),
            )
