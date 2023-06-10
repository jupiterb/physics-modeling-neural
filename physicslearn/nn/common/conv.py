from pydantic import BaseModel
from torch import nn
from typing import Sequence, Optional


class ConvNetworkConfig(BaseModel):
    channels: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Optional[Sequence[int]] = None
    paddings: Optional[Sequence[int]] = None
    transpose: bool = False


class ConvNetwork(nn.Module):
    def __init__(self, config: ConvNetworkConfig) -> None:
        super(ConvNetwork, self).__init__()
        self._transpose = config.transpose
        channels, kernel_sizes = config.channels, config.kernel_sizes
        default_ones = lambda: [1 for _ in kernel_sizes]
        strides = config.strides if config.strides is not None else default_ones()
        paddings = config.paddings if config.paddings is not None else default_ones()

        in_channels = config.channels[0]
        self._cnn = nn.Sequential()

        for out_channels, kernel_size, stride, padding in zip(
            channels[1:], kernel_sizes, strides, paddings
        ):
            conv_layer = self._get_conv_layer(
                in_channels, out_channels, kernel_size, stride, padding
            )
            conv_block = nn.Sequential(
                conv_layer, nn.BatchNorm2d(out_channels), nn.LeakyReLU()
            )
            self._cnn.append(conv_block)
            in_channels = out_channels

    def forward(self, X):
        return self._cnn(X)

    def _get_conv_layer(
        self,
        in_chnls: int,
        out_chnls: int,
        kernel: int,
        stride: int,
        padding: int,
    ) -> nn.Module:
        conv_class = nn.ConvTranspose2d if self._transpose else nn.Conv2d
        return conv_class(
            in_chnls,
            out_chnls,
            kernel,
            stride=stride,
            padding=padding,
        )
