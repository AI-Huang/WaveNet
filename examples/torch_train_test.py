#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-11-20 17:31
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

import torch
from wavenet.torch_fn.wavenet_lstm import WaveNet_LSTM


def main():
    # Load data
    raise NotImplementedError

    # Preparing model
    use_cuda = torch.cuda.is_available()  # CUDA
    if use_cuda:
        print("CUDA GPU available! Use GPU.")
    device = torch.device("cuda" if use_cuda else "cpu")

    seed = 42  # reproducibility, seed set to 42
    print(f"Set random seed to {seed} for model.")
    torch.manual_seed(seed)

    model = WaveNet_LSTM().to(device)


if __name__ == "__main__":
    main()
