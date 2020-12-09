#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-08-20 15:17
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

"""WaveNet implemented with PyTorch
Environments:
pytorch>=1.6.0
"""
import torch
from torch import nn


def lr_schedule(epoch):
    lr = 1e-4  # base learning rate
    if epoch >= 20:
        lr *= 0.1  # # reduced by 0.1 when finish training for 40 epochs
    return lr


class WaveNetLayer(nn.Module):
    """Single dilated conv layer in WaveNet
    # Arguments:
        x: input passed to this layer.
        filters: number of filters used for dilated convolution.
        kernel_size: the kernel size of the dilated convolution.
        dilation: the dilation rate for the dilated convolution.

    # Returns:
    """

    def __init__(self, filters, kernel_size, dilation):

        # Dilated Conv
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(1, filters, kernel_size,
                               padding=padding,
                               dilation=dilation)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x):
        conv_out = self.conv1(x)
        tanh_out = self.tanh(conv_out)
        sigm_out = self.sigm(conv_out)
        x_mul = torch.mul(tanh_out, sigm_out)
        # Skip connections
        x_skip_connection = self.conv2(x_mul)
        # Applying residual
        x_residual = torch.add(x, x_skip_connection)

        return x_residual, x_skip_connection


class WaveNetBlock(nn.Module):
    """wavenet_block, serveral wavenet layers which's dilation_rates are 2-based exponentially ascending, form a wavenet_block.
    # Arguments:
        x: input passed to this layer.
        filters: number of filters used for dilated convolution.
        kernel_size: the kernel size of the dilated convolution.
        dilation_rate: the dilation rate for the dilated convolution.

    # Returns:
    """

    def __init__(self, x, filters, kernel_size, n):
        self.n = n

        layers = []
        dilation_rates = [2**i for i in range(self.n)]
        for i, dilation_rate in enumerate(dilation_rates):
            name = "wavenet_layer_" + str(i)
            layers.append(
                (name, WaveNetLayer(filters, kernel_size, dilation_rate)))

        self.wavenet_layers = nn.Sequential(*layers)

    def forward(self, x):
        x_skip_connections = []

        # Apply dilated Conv
        for i in range(self.n):
            x, x_skip_connection = self.wavenet_layers[i](x)
            x_skip_connections.append(x_skip_connection)

        return x, x_skip_connections


def WaveNet(input_shape, filters, kernel_size, n):
    """WaveNet model. In this configuration, we follow the origin paper, extract skip_connection layers' output to produce predictions.
    # Arguments:
        input_shape:
        filters:
        kernel_size:
        n:

    # Returns:
    """
    # Apply causal conv to the input
    input_ = Input(shape=input_shape)
    x = nn.Conv1d(filters=filters,
                  kernel_size=1,
                  padding='same')(input_)

    x_residual, x_skip_connections = wavenet_block(x, filters, kernel_size, n)

    # Note that the x_residual output port is not used. It may be used to form multi wavenet_block in a cascading configuration.

    # Model top layers, including fully-connected layer, which produces output
    x_sum = Add()(x_skip_connections)
    x = Activation("relu")(x_sum)
    x = nn.Conv1d(1, 1, activation="relu")(x)
    x = nn.Conv1d(1, 1)(x)
    x = Flatten()(x)
    x = Dense(256, activation="softmax")(x)
    model = Model(inputs=[input_], outputs=[x])

    return model


def build_and_compile_model(input_shape, filters, kernel_size, n):
    """
    """
    model = WaveNet(input_shape, filters, kernel_size, n)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam",
        metrics=["accuracy"])
    return model


def WaveNet_LSTM(input_shape, activation=None, batch_norm=False, attention_type="custom"):
    """WaveNet_LSTM model. In this configuration, we use the x_residual port(s), connecting them in a cascading way, and pass the output to a LSTM layer.
    Inputs:
        input_shape:
        filters:
        kernel_size:
        n:
    Return:
    """
    # Model parameters
    filters = 16
    kernel_size = 3
    ns = [8, 5, 3]
    # Apply causal conv to the input
    input_ = Input(shape=input_shape)
    x = nn.Conv1d(filters=filters,
                  kernel_size=1,
                  padding='same')(input_)

    for i in range(3):
        # x_skip_connections is not used.
        x, _ = wavenet_block(x, filters, kernel_size, ns[i])
        if activation:
            x = Activation(activation)(x)
        x = AveragePooling1D(10)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    if attention_type == "official":
        x = Attention()([x, x])
    else:
        # x = myAttention(input_shape[0]//1000)(x)  # 150
        pass

    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(1)(x)
    model = Model(inputs=input_, outputs=x)

    return model


def main():
    pass


if __name__ == "__main__":
    main()
