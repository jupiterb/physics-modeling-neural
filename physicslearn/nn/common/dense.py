from pydantic import BaseModel
from torch import nn
from typing import Sequence


class DenseNetworkConfig(BaseModel):
    # could be more complex in the future
    layer_sizes: Sequence[int]


class DenseNetwork(nn.Module):
    def __init__(self, config: DenseNetworkConfig) -> None:
        super(DenseNetwork, self).__init__()
        self._mlp = nn.Sequential()
        input_size = config.layer_sizes[0]
        for output_size in config.layer_sizes[1:]:
            self._mlp.append(nn.Linear(input_size, output_size))
            self._mlp.append(nn.ReLU())
            input_size = output_size

    def forward(self, X):
        return self._mlp(X)
