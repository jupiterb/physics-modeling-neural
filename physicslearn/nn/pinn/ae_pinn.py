import numpy as np
import torch as th
from torch import nn
from typing import Sequence

from physicslearn.nn.common.conv import ConvNetwork, ConvNetworkConfig
from physicslearn.nn.common.dense import DenseNetwork, DenseNetworkConfig
from physicslearn.nn.pinn.abstract import PhysicsInformedNN


class PhysicsInformedAutoEncoder(PhysicsInformedNN):
    """A symmetric autoencoder that adds an additional time context to the encoder output"""

    def __init__(
        self,
        input_shape: Sequence[int],
        encoder_conv_config: ConvNetworkConfig,
        encoder_dense_config: DenseNetworkConfig,
        parameter_inference_config: DenseNetworkConfig,
    ) -> None:
        super(PhysicsInformedAutoEncoder, self).__init__()

        # build encoder
        encoder_conv = ConvNetwork(encoder_conv_config)
        encoder_dense = DenseNetwork(encoder_dense_config)

        # additional dense layer to connect conv encoder with dense encoder
        sample = th.ones((1, 1, *input_shape))
        encoder_conv_output_shape = encoder_conv(sample).shape[1:]
        encoder_conv_output_size = np.prod(encoder_conv_output_shape)
        encoder_conenct_layer_sizes = [
            encoder_conv_output_size,
            encoder_dense_config.layer_sizes[0],
        ]
        encoder_conenct_config = DenseNetworkConfig(
            layer_sizes=encoder_conenct_layer_sizes
        )
        encoder_connect = DenseNetwork(encoder_conenct_config)

        # full encoder
        self._encoder = nn.Sequential(
            encoder_conv, nn.Flatten(), encoder_connect, encoder_dense
        )

        # build decoder dense
        decoder_dense_layer_sizes = [
            size for size in reversed(encoder_dense_config.layer_sizes)
        ]
        # concat with time context
        decoder_dense_layer_sizes[0] += 1
        # add connect layer
        decoder_dense_layer_sizes.append(encoder_conv_output_size)
        decoder_dense_config = DenseNetworkConfig(layer_sizes=decoder_dense_layer_sizes)

        decoder_dense = DenseNetwork(decoder_dense_config)

        # build decoder conv
        decoder_conv_channels = [
            channels for channels in reversed(encoder_conv_config.channels)
        ]
        decoder_conv_config = encoder_conv_config.copy(deep=True)
        decoder_conv_config.channels = decoder_conv_channels
        decoder_conv_config.transpose = True

        decoder_conv = ConvNetwork(decoder_conv_config)

        # full decoder
        self._decoder = nn.Sequential(
            decoder_dense,
            nn.Unflatten(dim=1, unflattened_size=encoder_conv_output_shape),
            decoder_conv,
        )

        # base parameter inference network
        parameter_inference_base = DenseNetwork(parameter_inference_config)

        # connect with encoder
        parameter_inference_connect_layer_sizes = [
            encoder_dense_config.layer_sizes[-1] + 1,
            parameter_inference_config.layer_sizes[0],
        ]
        parameter_inference_connect_config = DenseNetworkConfig(
            layer_sizes=parameter_inference_connect_layer_sizes
        )
        parameter_inference_connect = DenseNetwork(parameter_inference_connect_config)

        # full parameter inference network
        self._parameter_inference = nn.Sequential(
            parameter_inference_connect, parameter_inference_base
        )

    def forward(self, U: th.Tensor, T: th.Tensor):
        U_encoded: th.Tensor = self._encoder(U)
        U_encoded_in_time = th.cat((U_encoded, T), 1)
        U_decoded = self._decoder(U_encoded_in_time)
        parameters = self._parameter_inference(U_encoded_in_time)
        return U_decoded, parameters
