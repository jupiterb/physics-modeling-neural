from abc import ABC, abstractmethod
import numpy as np
import torch as th
from typing import Optional, Union, Sequence


Array = Union[np.ndarray, th.Tensor]


def _to_numpy(array: Array):
    if isinstance(array, np.ndarray):
        return array
    else:
        return array.detach().numpy()


class PhyDiffEq(ABC):
    @abstractmethod
    def _du_dt(
        self, U: np.ndarray, T: np.ndarray, parameters: Optional[np.ndarray]
    ) -> np.ndarray:
        raise NotImplementedError()

    def __call__(
        self, U: Array, T: Array, parameters: Optional[Array] = None
    ) -> np.ndarray:
        """
        Computes differential equation

        Parameters:
            U: states, tensor of shape (batch size, *state shape)
            T: time, tensor of shape (batch size, 1)
            parameters (of equation): optional tensor of shape (batch size, number of parameters)

        Returns:
            dU/dT
        """
        return self._du_dt(
            _to_numpy(U),
            _to_numpy(T),
            _to_numpy(parameters) if parameters is not None else None,
        )
