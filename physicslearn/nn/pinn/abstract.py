from abc import ABC, abstractmethod
import torch as th


class PhysicsInformedNN(th.nn.Module, ABC):
    @abstractmethod
    def forward(self, U: th.Tensor, T: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Returns (dU/dT, parameters of equation) for whole batch"""
        raise NotImplementedError
