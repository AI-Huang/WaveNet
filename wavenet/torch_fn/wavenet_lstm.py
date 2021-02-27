#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-10-20 17:52
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

"""WaveNet_LSTM implemented with PyTorch
Environments:
pytorch>=1.6.0
"""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from wavenet.torch_fn.wavenet import WaveNetBlock


def lr_schedule(epoch):
    lr = 1e-4  # base learning rate
    if epoch >= 20:
        lr *= 0.1  # # reduced by 0.1 when finish training for 40 epochs
    return lr


class WaveNet_LSTM(nn.Module):
    """WaveNet_LSTM model.
    # Arguments:
        input_shape:
        activation:
        batch_norm:
        attention_type:

    # Returns:
    """

    def __init__(self, input_size, activation=None, batch_norm=False, attention_type="custom"):
        super().__init__()

        self.activation = activation
        self.batch_norm = batch_norm

        # Model parameters
        out_channels = 16
        kernel_size = 3
        self.ns = [8, 5, 3]
        self.conv1 = nn.Conv1d(1, out_channels, 1)

        layers = []
        for i, n in enumerate(self.ns):
            in_channels = out_channels
            layers.append(
                ("wavenet_block_" + str(i),
                 WaveNetBlock(in_channels, out_channels, kernel_size, n))
            )
        self.wavenet_blocks = nn.Sequential(OrderedDict(layers))

        self.lstm = nn.LSTM(input_size=input_size//1000,
                            hidden_size=64, bidirectional=True)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply causal conv to the input
        x = self.conv1(x)

        # Note that the x_residual output port is not used. It may be used to form multi wavenet_block in a cascading configuration.
        for i, _ in enumerate(self.ns):
            x, _ = self.wavenet_blocks[i](x)
            # if activation:
            # x = Activation(activation)(x)
            x = F.avg_pool1d(x, 10)

        x, (hn, cn) = self.lstm(x)
        # if attention_type == "official":
        # x = Attention()([x, x])
        x = x.view(-1, x.size()[1:].numel())

        x = F.dropout(x, 0.2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
