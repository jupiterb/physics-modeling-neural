import numpy as np
from typing import Optional

from physicslearn.equations.abstract import PhyDiffEq


class LimitSpaceWrapper(PhyDiffEq):
    def __init__(self, eq: PhyDiffEq, space_limit: np.ndarray) -> None:
        self._eq = eq
        self._space_limit = space_limit

    def _du_dt(
        self, U: np.ndarray, T: np.ndarray, parameters: Optional[np.ndarray]
    ) -> np.ndarray:
        return self._eq._du_dt(U, T, parameters) * self._space_limit


class ClipStateWrapper(PhyDiffEq):
    def __init__(self, eq: PhyDiffEq, min: float, max: float) -> None:
        self._eq = eq
        self._min = min
        self._max = max

    def _du_dt(
        self, U: np.ndarray, T: np.ndarray, parameters: Optional[np.ndarray]
    ) -> np.ndarray:
        du_dt = self._eq._du_dt(U, T, parameters)
        next_U = U + du_dt

        next_U[next_U > self._max] = self._max
        next_U[next_U < self._min] = self._min

        return next_U - U


class ScaleWrapper(PhyDiffEq):
    def __init__(self, eq: PhyDiffEq, scale: float) -> None:
        self._eq = eq
        self._scale = scale

    def _du_dt(
        self, U: np.ndarray, T: np.ndarray, parameters: Optional[np.ndarray]
    ) -> np.ndarray:
        u_rescaled = U / self._scale
        du_dt = self._eq(u_rescaled, T, parameters)
        du_dt *= self._scale
        return du_dt
