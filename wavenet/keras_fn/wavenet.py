#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-04-20 17:07
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

"""WaveNet implemented with Keras functional API
Environments:
tensorflow>=2.1.0
"""
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv1D, Multiply, Add, Flatten, Dense


def lr_schedule(epoch):
    lr = 1e-4  # base learning rate
    if epoch >= 20:
        lr *= 0.1  # # reduced by 0.1 when finish training for 40 epochs
    return lr


def wavenet_layer(x, filters, kernel_size, dilation_rate):
    """Single dilated conv layer in WaveNet
    Inputs:
        x: input passed to this layer.
        filters: number of filters used for dilated convolution.
        kernel_size: the kernel size of the dilated convolution.
        dilation_rate: the dilation rate for the dilated convolution.
    """
    # Cache the input
    x_residual = x
    # Dilated Conv
    tanh_out = Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      activation='tanh',
                      dilation_rate=dilation_rate)(x)
    sigm_out = Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      activation='sigmoid',
                      dilation_rate=dilation_rate)(x)
    x = Multiply()([tanh_out, sigm_out])
    # Skip connections
    x_skip_connection = Conv1D(1, 1)(x)
    # Applying residual
    x_residual = Add()([x_residual, x_skip_connection])

    return x_residual, x_skip_connection


def wavenet_block(x, filters, kernel_size, n):
    """wavenet_block, serveral wavenet layers which's dilation_rates are 2-based exponentially ascending, form a wavenet_block.
    Inputs:
        x: input passed to this layer.
        filters: number of filters used for dilated convolution.
        kernel_size: the kernel size of the dilated convolution.
        dilation_rate: the dilation rate for the dilated convolution.
    """
    x_skip_connections = []

    # Apply dilated Conv
    dilation_rates = [2**i for i in range(n)]
    for dilation_rate in dilation_rates:
        x, x_skip_connection = wavenet_layer(
            x, filters, kernel_size, dilation_rate)
        x_skip_connections.append(x_skip_connection)

    return x, x_skip_connections


def WaveNet(input_shape, filters, kernel_size, n):
    """WaveNet model. In this configuration, we follow the origin paper, extract skip_connection layers' output to produce predictions.
    Inputs:
        input_shape:
        filters:
        kernel_size:
        n:
    Return:
    """
    # Apply causal conv to the input
    input_ = Input(shape=input_shape)
    x = Conv1D(filters=filters,
               kernel_size=1,
               padding='same')(input_)

    x_residual, x_skip_connections = wavenet_block(x, filters, kernel_size, n)

    # Note that the x_residual output port is not used. It may be used to form multi wavenet_block in a cascading configuration.

    # Model top layers, including fully-connected layer, which produces output
    x_sum = Add()(x_skip_connections)
    x = Activation("relu")(x_sum)
    x = Conv1D(1, 1, activation="relu")(x)
    x = Conv1D(1, 1)(x)
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


def main():
    pass


if __name__ == "__main__":
    main()
