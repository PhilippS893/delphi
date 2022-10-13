import torch.nn as nn

def convblock3d(
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        conv_rep: int=1,
        pooling_kernel_size: int=2,
        add_batch_norm: bool=True,
        add_pooling: bool=True,
        activ_fn: str="ReLU",
) -> nn.Sequential:
    '''Returns a configuered 3D convolutional block.

    Parameters
    ----------
    in_channels: :py:obj:`ìnt`
        The input channels to the convolutional layer.
    out_channels: :py:obj:`int`
        The output channels of the convolutional layer.
    kernel_size: :py:obj:`int`
        The kernel size of each filter of the convolutional layer. (optional, default=3)
    conv_rep: :py:obj:`int`
        The number of repetitions of the same convolutional layer. Note that additoinal layers have out_channels as
        input and output (optional, default=1)
    pooling_kernel_size: :py:obj:`int`
        The kernel size of the pooling layer (optional, default=2)
    add_batch_norm: :py:obj:`bool`
        Conditional to add a batch normalization layer after all convolutional layers but before ReLU
        (optional, default=``True``)
    add_pooling: :py:obj:`bool`
        Conditional to add a MaxPooling layer after all layers (optional, default=``True``)
    activ_fn: :py:obj:`str` ToDo:make this into a torch.nn.activation class or sth.
        The activation function to use. (Not implemented yet)

    Returns
    -------
    layer_block: :py:obj:`torch.nn.Sequential`
        The block of sequential layers.

    '''
    layer_block = nn.Sequential()

    # add the conv3d layer
    for i in range(conv_rep):
        layer_block.append(
            nn.Conv3d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2    # this keeps the dimensions of the output equal to the input
            )
        )

    # add the batchnorm layer after the convlayer(s)
    if add_batch_norm:
        layer_block.append(
            nn.BatchNorm3d(out_channels)
        )

    # add the activation function
    layer_block.append(
        nn.ReLU(inplace=False)  # we need this to be False to do LRP. Inplace=True led to issues
    )

    # add the pooling layer if wanted
    if add_pooling:
        layer_block.append(
            nn.MaxPool3d(kernel_size=pooling_kernel_size)
        )

    return layer_block

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
