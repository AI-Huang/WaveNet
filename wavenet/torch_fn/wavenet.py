#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-08-20 15:17
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

"""WaveNet implemented with PyTorch
Environments:
pytorch>=1.6.0
"""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


class WaveNetLayer(nn.Module):
    """Single dilated conv layer in WaveNet
    # Arguments:
        x: input passed to this layer.
        out_channels: number of out_channels used for dilated convolution.
        kernel_size: the kernel size of the dilated convolution.
        dilation: the dilation rate for the dilated convolution.

    # Returns:
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Dilated Conv
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding,
                               dilation=dilation)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=1, kernel_size=1)

    def forward(self, x):
        conv_out = self.conv1(x)

        tanh_out = self.tanh(conv_out)
        sigm_out = self.sigm(conv_out)

        x_mul = torch.mul(tanh_out, sigm_out)
        x_skip_connection = self.conv2(x_mul)
        x_residual = torch.add(x, x_skip_connection)

        return x_residual, x_skip_connection


class WaveNetBlock(nn.Module):
    """wavenet_block, serveral wavenet layers which's dilation_rates are 2-based exponentially ascending, form a wavenet_block.
    # Arguments:
        out_channels: number of out_channels used for dilated convolution.
        kernel_size: the kernel size of the dilated convolution.
        n: number of the dilated convolution layers.

    # Returns:
    """

    def __init__(self, in_channels, out_channels, kernel_size, n):
        super().__init__()
        self.n = n

        layers = []
        dilation_rates = [2**i for i in range(self.n)]
        for i, dilation_rate in enumerate(dilation_rates):
            layers.append(
                ("wavenet_layer_" + str(i),
                 WaveNetLayer(in_channels, out_channels, kernel_size, dilation_rate))
            )
        self.wavenet_layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x_skip_connections = []

        # Apply dilated Conv
        for i in range(self.n):
            x, x_skip_connection = self.wavenet_layers[i](x)
            x_skip_connections.append(x_skip_connection)

        return x, x_skip_connections


class WaveNet(nn.Module):
    """WaveNet model. In this configuration, we follow the origin paper, extract skip_connection layers' output to produce predictions.
    # Arguments:
        input_size:
        out_channels:
        kernel_size:
        n:

    # Returns:
    """

    def __init__(self, input_size, out_channels, kernel_size, n):
        self.in_channels = 1

        self.conv1 = nn.Conv1d(1, out_channels, 1)
        self.wavenet_block = WaveNetBlock(
            self.in_channels, out_channels, kernel_size, n)
        self.conv2 = nn.Conv1d(1, 1, 1)
        self.conv3 = nn.Conv1d(1, 1, 1)
        self.fc = nn.Linear(input_size, 256)

    def forward(self, x):
        """
        docstring
        """
        # Apply causal conv to the input
        x = self.conv1(x)

        # Note that the x_residual output port is not used. It may be used to form multi wavenet_block in a cascading configuration.
        x_residual, x_skip_connections = self.wavenet_block(x)

        # Model top layers, including fully-connected layer, which produces output
        x_sum = torch.sum(x_skip_connections, dim=1)
        x = F.relu(x_sum)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.softmax(self.fc(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    pass


if __name__ == "__main__":
    main()
