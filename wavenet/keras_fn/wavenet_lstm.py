#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-10-20 17:52
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

"""WaveNetLSTM model implemented with Keras functional API
Environments:
tensorflow>=2.1.0
"""
from wavenet.keras_fn.wavenet import wavenet_block
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv1D, AveragePooling1D, Bidirectional, LSTM, Dropout, Dense, Attention


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
    x = Conv1D(filters=filters,
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
