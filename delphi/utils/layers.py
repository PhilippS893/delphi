import torch.nn as nn


def convbatchrelu2d(in_channels: int, out_channels: int, kernel_size: int, pooling_kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(pooling_kernel)
    )


def convbatchrelu3d(in_channels: int, out_channels: int, kernel_size: int, pooling_kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.ReLU(inplace=False),
        nn.BatchNorm3d(out_channels),
        nn.MaxPool3d(pooling_kernel)
    )


def linrelulayer(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=False)
    )
