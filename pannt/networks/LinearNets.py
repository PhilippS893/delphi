from pannt.networks.Base import TemplateModel
from pannt.utils.tools import read_config
from pannt.utils.train_fns import standard_train
from typing import List, Optional
import torch.nn as nn
from pannt.utils.layers import linrelulayer
import torch
import os


class SimpleLinearModel(TemplateModel):
    _REQUIRED_PARAMS: List[Optional[str]] = ["lin_neurons"]

    def _use_default_config(self):
        return {
            'lin_neurons': [128, 64, 32]
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
        super(SimpleLinearModel, self).__init__()
        print('Loading from config file %s/config.yaml' % path_to_config_file)
        self.config = read_config(os.path.join(path_to_config_file, "config.yaml"))
        self.train_fn = self.config['train_fn']
        self._setup_layers()
        self.load_state_dict(torch.load(os.path.join(path_to_config_file, "state_dict.pth")))

    def __init__(
            self,
            input_vals: int,
            n_classes: int,
            config: dict = None,
    ):
        super(SimpleLinearModel, self).__init__()
        self._init(input_vals=input_vals, n_classes=n_classes, config=config)

    def __init__(
            self,
            input_vals: int,
            n_classes: int,
            train_fn: type(standard_train) = standard_train,
    ):
        super(SimpleLinearModel, self).__init__(train_fn=train_fn)
        self._init(input_vals=input_vals, n_classes=n_classes)

    def __init__(
            self,
            input_vals: int,
            n_classes: int,
            config: dict = None,
            train_fn: type(standard_train) = standard_train,
    ):
        super(SimpleLinearModel, self).__init__(train_fn=train_fn)
        self._init(input_vals=input_vals, n_classes=n_classes, config=config)

    def _init(self, input_vals: int, n_classes: int, config: dict = None):
        self.config = self._use_default_config() if config is None else self._update_params_in_config(config)
        self._check_params_in_config()

        self.config['input_vals'] = input_vals
        self.config['n_classes'] = n_classes
        self.config['train_fn'] = self.train_fn

        self._setup_layers()

    def _setup_layers(self):
        r"""

        :return:
        """

        self.lin = nn.Sequential()
        for n in range(len(self.config['lin_neurons'])):
            if n == 0:
                self.lin.add_module(
                    f'lin{n}',
                    linrelulayer(
                        in_features=self.config['input_vals'], out_features=self.config['lin_neurons'][n]
                    )
                )
            else:
                self.lin.add_module(
                    f'lin{n}',
                    linrelulayer(
                        in_features=self.config['lin_neurons'][n - 1], out_features=self.config['lin_neurons'][n]
                    )
                )

        self.out = nn.Linear(self.config['lin_neurons'][-1], self.config['n_classes'])

        self.dropout = nn.Dropout(p=.5)
        self.SM = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.view(-1, self.config['input_vals'])  # flatten the input in case they are multidimensional

        # pass the input sequentially through all layers
        for i, layer in enumerate(self.lin):
            if i < len(self.lin) - 1:
                x = layer(x)
                x = self.dropout(x)

        # use dropout for the last layer
        x = self.lin[-1](x)
        return self.out(x)
