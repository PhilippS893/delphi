from delphi.networks.Base import TemplateModel
from delphi.utils.train_fns import standard_train
from delphi.utils.tools import read_config, get_cnn_output_dim, get_maxpool_output_dim
import os
import torch
from typing import List, Optional
import torch.nn as nn
import numpy as np
from delphi.utils.layers import convbatchrelu2d, convbatchrelu3d, linrelulayer, convblock3d


def _get_dims_of_last_convlayer(input_dims, n_layers, cnn_kernel_size, pooling_kernel_size):
    r"""

    Args:
        input_dims:
        n_layers:
        cnn_kernel_size:
        pooling_kernel_size:

    Returns:

    """
    from delphi.utils.tools import get_maxpool_output_dim, get_cnn_output_dim
    for i in range(n_layers - 1):
        input_dims = get_cnn_output_dim(input_dims, cnn_kernel_size[i], cnn_kernel_size[i] // 2, 1)
        input_dims = get_maxpool_output_dim(input_dims, pooling_kernel_size, 0, pooling_kernel_size, 1)

    return input_dims


class FCN3d(TemplateModel):

    def _use_default_config(self) -> dict:
        default_cfg = {
            'n_hidden_layers': 4,
            'input_channels': 1,
            'channels': [8, 16, 32, 64],
            'kernel_size': 5,
            'pooling_kernel': 2,
            'dropout': .5,
        }

        n_layers = default_cfg["n_hidden_layers"]

        default_cfg["add_pooling"] = [True] * n_layers
        default_cfg["add_batchnorm"] = [True] * n_layers
        default_cfg["add_dropout"] = [True] * n_layers

        # for more flexibility in coding, check if the following variables are scalars
        # if so, multiply them by n_layers
        potential_scalars = ["kernel_size", "pooling_kernel", "dropout"]
        for i, var in enumerate(potential_scalars):
            if isinstance(default_cfg[var], int) or isinstance(default_cfg[var], float):
                default_cfg[var] = [default_cfg[var]] * n_layers

        return default_cfg

    def __init__(
            self,
            input_dims: tuple,
            n_classes: int,
            config: dict = None,
    ):
        super().__init__()
        self.config = self._use_default_config() if config is None else self._update_params_in_config(config)
        self.config["input_dims"] = input_dims
        self.config["n_classes"] = n_classes
        self.SM = torch.nn.Softmax(dim=1)
        self.model = self._setup_layers()

    def _setup_layers(self):
        # we accumulate layers in layer_stack by layer_stack.extend([<layers>])
        layer_stack = []

        out_shape = self.config["input_dims"]
        for i in range(self.config["n_hidden_layers"]):
            convblock3d(
                layer_stack,
                in_channels=self.config["input_channels"] if i == 0 else self.config["channels"][i - 1],
                out_channels=self.config["channels"][0] if i == 0 else self.config["channels"][i],
                kernel_size=self.config["kernel_size"][i],
                conv_rep=1 if "conv_rep" not in self.config else self.config["conv_rep"][i],
                pooling_kernel_size=2 if "pooling_kernel" not in self.config else self.config["pooling_kernel"][i],
                activ_fn="ReLU",
                add_pooling=True if "add_pooling" not in self.config else self.config["add_pooling"][i],
                add_batch_norm=True if "add_batchnorm" not in self.config else self.config["add_batchnorm"][i],
                conv_kwargs={} if "conv_kwargs" not in self.config else self.config["conv_kwargs"]
            ),

            if "add_dropout" in self.config:
                if self.config["add_dropout"][i]:
                    layer_stack.extend([
                        nn.Dropout(p=self.config["dropout"][i])
                    ])

            # compute the layers output shape
            out_shape = get_cnn_output_dim(
                input_dims=out_shape,
                kernel_size=self.config["kernel_size"][i],
                **self.config["conv_kwargs"]
            )

            use_pooling = True if "add_pooling" not in self.config else self.config["add_pooling"][i]
            # compute the pooling layers output shape
            if use_pooling:
                out_shape = get_maxpool_output_dim(out_shape, self.config["pooling_kernel"][i], 0,
                                                   self.config["pooling_kernel"][i], 1)

        # add the output linear layer
        layer_stack.extend([
            nn.Flatten(),
            nn.Linear(
                in_features=np.product(out_shape) * self.config["channels"][self.config["n_hidden_layers"] - 1],
                out_features=self.config["n_classes"]
            )
        ])

        return torch.nn.Sequential(*layer_stack)

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input)


class BrainStateClassifier3d(TemplateModel):
    _REQUIRED_PARAMS: List[Optional[str]] = ["channels", "kernel_size", "pooling_kernel", "lin_neurons", "dropout"]

    def _use_default_config(self) -> dict:
        return {
            'channels': [1, 8, 16, 32, 64],
            'kernel_size': 5,
            'pooling_kernel': 2,
            'lin_neurons': [128, 64],
            'dropout': .5,
        }

    def __init__(
            self,
            path_to_config_file: str,
    ):
        r"""
        If the constructor is called with a path then we assume that the network was pretrained. This is particularly
        interesting for transfer learning approaches.

        :param path_to_config_file:
        """
        super(BrainStateClassifier3d, self).__init__()
        print('Loading from config file %s/config.yaml' % path_to_config_file)
        self.config = read_config(os.path.join(path_to_config_file, "config.yaml"))
        self.train_fn = self.config['train_fn']  # override the default
        self._setup_layers()
        self.load_state_dict(
            torch.load(os.path.join(path_to_config_file, "state_dict.pth"), map_location=torch.device("cpu")))

    def __init__(
            self,
            input_dims: tuple,
            n_classes: int,
            config: dict = None,
    ):
        super(BrainStateClassifier3d, self).__init__()
        self._init(input_dims=input_dims, n_classes=n_classes, config=config)

    def __init__(
            self,
            input_dims: tuple,
            n_classes: int,
            train_fn: type(standard_train) = standard_train,
    ):
        super(BrainStateClassifier3d, self).__init__(train_fn=train_fn)
        self._init(input_dims=input_dims, n_classes=n_classes)

    def __init__(
            self,
            input_dims: tuple,
            n_classes: int,
            config: dict = None,
            train_fn: type(standard_train) = standard_train,
    ):
        r"""
        Constructor to build a simple 3d Convolutional Neural Network classifier.

        Args:
            input_dims:
            n_classes:
            train_fn:
            config:
        """

        super(BrainStateClassifier3d, self).__init__(train_fn=train_fn)
        self._init(input_dims=input_dims, n_classes=n_classes, config=config)

    def _init(self, input_dims: tuple, n_classes: int, config: dict = None):
        self.config = self._use_default_config() if config is None else self._update_params_in_config(config)
        self._check_params_in_config()

        if isinstance(self.config["kernel_size"], int):
            self.config["kernel_size"] = [self.config["kernel_size"] for i in
                                          range(0, len(self.config["channels"]) - 1)]

        self.config['input_dims'] = input_dims
        self.config['n_classes'] = n_classes
        self.config['train_fn'] = self.train_fn

        self._setup_layers()

    def forward(self, x):
        """
        forward pass through the network
        :param x:   input to the network
        :return:    vector with n_elements = n_classes
        """

        # pass the input sequentially through all convlayers first
        for i, convlayer in enumerate(self.conv):
            x = convlayer(x)

        # flatten the output of the last convlayer (linlayers want a 1D input)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # pass the flattened input through n-1 linlayers followed by a dropout
        for i, linlayer in enumerate(self.lin):
            if i < len(self.lin) - 1:
                x = linlayer(x)
                x = self.dropout(x)

        # pass the input through the last linlayer but do not use dropout!
        x = self.lin[-1](x)

        # classify
        x = self.out(x)

        return x

    def _setup_layers(self):
        r"""

        Returns:

        """
        self.dropout = nn.Dropout(p=self.config['dropout'])
        self.relu = nn.ReLU(inplace=False)  # inplace True leads to issues in the LRP algorithm.
        self.SM = nn.Softmax(dim=1)

        # compute the input dimensions of the last convolutional layer.
        self.config['last_cnn_dims'] = _get_dims_of_last_convlayer(
            self.config['input_dims'],
            len(self.config['channels']),
            self.config['kernel_size'],
            self.config['pooling_kernel']
        ).tolist()

        # add convolutional layers as configured by the function convbatchrelu3d
        self.conv = nn.Sequential()
        for n in range(len(self.config['channels']) - 1):
            self.conv.add_module(
                'convlayer%d' % n,
                convbatchrelu3d(
                    self.config['channels'][n],  # input channels
                    self.config['channels'][n + 1],  # output channels
                    self.config['kernel_size'][n],
                    self.config['pooling_kernel']
                )
            )

        # add linear layers as configured by the function linrelulayer
        self.lin = nn.Sequential()
        for n in range(len(self.config['lin_neurons'])):
            if n == 0:
                self.lin.add_module(
                    'lin%d' % n,
                    linrelulayer(
                        self.config['channels'][-1] * np.product(self.config['last_cnn_dims']),
                        self.config['lin_neurons'][n]
                    )
                )
            else:
                self.lin.add_module(
                    'lin%d' % n,
                    linrelulayer(
                        self.config['lin_neurons'][n - 1],
                        self.config['lin_neurons'][n]
                    )
                )

        # add the output layer
        self.out = nn.Linear(self.config['lin_neurons'][-1], self.config['n_classes'])


class Simple2dCnnClassifier(TemplateModel):
    _REQUIRED_PARAMS: List[Optional[str]] = ["channels", "kernel_size", "pooling_kernel", "lin_neurons", "dropout"]

    def __init__(
            self,
            path_to_config_file: str,
    ):
        r"""
        If the constructor is called with a path then we assume that the network was pretrained. This is particularly
        interesting for transfer learning approaches.

        :param path_to_config_file:
        """
        super(Simple2dCnnClassifier, self).__init__()
        print('Loading from config file %s/config.yaml' % path_to_config_file)
        self.config = read_config(os.path.join(path_to_config_file, "config.yaml"))
        self.train_fn = self.config['train_fn']  # override the default
        self._setup_layers()
        self.load_state_dict(torch.load(os.path.join(path_to_config_file, "state_dict.pth")))

    def __init__(
            self,
            input_dims: tuple,
            n_classes: int,
            config: dict = None,
    ):
        super(Simple2dCnnClassifier, self).__init__()
        self._init(input_dims=input_dims, n_classes=n_classes, config=config)

    def __init__(
            self,
            input_dims: tuple,
            n_classes: int,
            train_fn: type(standard_train) = standard_train,
    ):
        super(Simple2dCnnClassifier, self).__init__(train_fn=train_fn)
        self._init(input_dims=input_dims, n_classes=n_classes)

    def __init__(
            self,
            input_dims: tuple,
            n_classes: int,
            config: dict = None,
            train_fn: type(standard_train) = standard_train,
    ):
        r"""
        Constructor to build a simple 2d Convolutional Neural Network classifier.

        Args:
            input_dims:
            n_classes:
            train_fn:
            config:
        """

        super(Simple2dCnnClassifier, self).__init__(train_fn)
        self._init(input_dims=input_dims, n_classes=n_classes, config=config)

    def _init(self, input_dims: tuple, n_classes: int, config: dict = None):
        self.config = self._use_default_config() if config is None else self._update_params_in_config(config)
        self._check_params_in_config()

        if isinstance(self.config["kernel_size"], int):
            self.config["kernel_size"] = [self.config["kernel_size"] for i in
                                          range(0, len(self.config["channels"]) - 1)]

        self.config['input_dims'] = input_dims
        self.config['n_classes'] = n_classes
        self.config['train_fn'] = self.train_fn

        self._setup_layers()

    def forward(self, x):
        """
        forward pass through the network
        :param x:   input to the network
        :return:    vector with n_elements = n_classes
        """

        # pass the input sequentially through all convlayers first
        for i, convlayer in enumerate(self.conv):
            x = convlayer(x)

        # reshape the output of the last convlayer (linlayers want a 1D input)
        # x.size(0) provides the number of exemplars in a batch
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # pass the reshaped input through n-1 linlayers followed by a dropout
        for i, linlayer in enumerate(self.lin):
            if i < len(self.lin) - 1:
                x = linlayer(x)
                x = self.dropout(x)

        # pass the input through the last linrelulayer but do not use dropout!
        x = self.lin[-1](x)

        # classify
        x = self.out(x)

        return x

    def _setup_layers(self):
        r"""

        :return:
        """

        self.relu = nn.ReLU(inplace=False)  # using inplace=True can lead to issues with LRP
        self.dropout = nn.Dropout(self.config['dropout'])
        self.config['last_cnn_dims'] = _get_dims_of_last_convlayer(
            self.config['input_dims'],
            len(self.config['channels']),
            self.config['kernel_size'],
            self.config['pooling_kernel']
        ).tolist()

        # construct the layers
        # add convolutional layers as configured by the function utils.layers.convbatchrelu3d
        self.conv = nn.Sequential()
        for n in range(len(self.config['channels']) - 1):
            self.conv.add_module(
                'convlayer%d' % n,
                convbatchrelu2d(
                    in_channels=self.config['channels'][n], out_channels=self.config['channels'][n + 1],
                    kernel_size=self.config['kernel_size'][n], pooling_kernel=self.config['pooling_kernel']
                )
            )

        # add linear layers as configured by the function utils.layers.linrelulayer
        self.lin = nn.Sequential()
        for n in range(len(self.config['lin_neurons'])):
            if n == 0:
                self.lin.add_module(
                    'lin%d' % n,
                    linrelulayer(
                        in_features=self.config['channels'][-1] * np.product(self.config['last_cnn_dims']),
                        out_features=self.config['lin_neurons'][n]
                    )
                )
            else:
                self.lin.add_module(
                    'lin%d' % n,
                    linrelulayer(
                        in_features=self.config['lin_neurons'][n - 1],
                        out_features=self.config['lin_neurons'][n]
                    )
                )

        # add the output layer
        self.out = nn.Linear(in_features=self.config['lin_neurons'][-1], out_features=self.config['n_classes'])

        # soft max layer to get probabilities for predicted classes
        self.SM = nn.Softmax(dim=1)

    def _use_default_config(self) -> dict:
        return {
            'channels': [1, 8, 16, 32],
            'kernel_size': 5,
            'pooling_kernel': 2,
            'lin_neurons': [256, 128],
            'dropout': .5,
        }
