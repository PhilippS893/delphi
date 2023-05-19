import copy
import os
import torch.nn
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Callable, List, Optional
from delphi.utils.tools import read_config
from torch.utils.data import DataLoader
from delphi.utils.MultiMethod import MultipleMeta
from delphi.utils.train_fns import standard_train
import yaml


def metaclass_resolver(*classes):
    metaclass = tuple(set(type(cls) for cls in classes))
    metaclass = metaclass[0] if len(metaclass) == 1 \
        else type("_".join(mcls.__name__ for mcls in metaclass), metaclass, {})  # class M_C
    return metaclass("_".join(cls.__name__ for cls in classes), classes, {})


class A(metaclass=MultipleMeta):
    pass


class B(ABC):
    pass


class C(metaclass_resolver(A, B)):
    pass


class Model(C):
    r"""
    Abstract Class to define abstract methods for flexibility in creating networks.
    """

    @abstractmethod
    def fit(self, train_data: DataLoader, **kwargs):
        r"""
        Override this method in your custom network class or use the TemplateModel class to actually train your model.

        Args:
            train_data (DataLoader):
                The data to train on
        Returns:
            statistics about training, e.g.  training loss and accuracy, etc.
        """
        pass


class TemplateModel(nn.Module, Model):
    r"""
    Template Model Class that implements the 'fit' method. If a class inherits from this class the fit method calls the
    user supplied training method.
    """

    _REQUIRED_PARAMS: List[Optional[str]] = None

    @abstractmethod
    def _use_default_config(self) -> dict:
        r"""
        Use default parameters set by the inheriting class if the user does not supply their own configuration.

        Returns:
            namedtuple of the default parameters for the network class.
        """
        pass

    def __init__(self, train_fn: Callable = standard_train, config=None):
        r"""
        Constructor for the TemplateModel
        Args:
            train_fn:
            config:
        """
        super(TemplateModel, self).__init__()

        self.config = config
        self.train_fn = train_fn

    def load(self, model_name: str) -> None:
        r"""
        loads the config and parameters of the provided model_name. Loaded model can then be used, e.g.,
        for transfer learning

        Args:
            model_name: the path/model_name to the networks config and pth file. Important: OMIT the file ending!

        """

        config = read_config(os.path.join(model_name, "config.yaml"))
        # config = namedtuple("config", hps.keys())(*hps.values())
        self.__init__(config=config)  # reinitialize the model with the config supplied by the .yaml file
        self.load_state_dict(torch.load(os.path.join(model_name, "state_dict.pth")))

    def save(self, save_dir: str, save_full=False) -> None:
        r"""
        saves the statedict and the configuration of the network.
        The resulting .pth and .yaml file can be loaded for later use.

        Args:
            save_dir (str): path with filename for the <save_name>.pth and <save_name>_config.yaml
            save_full (bool): flag if the entire model should be saved instead of only the state_dict.
        """

        # in case a path is provided (some/path/model_name) check if the parent directories exist. If not, create them.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_full:
            print("Saving entire model %s/model.pth" % save_dir)
            torch.save(self, os.path.join(save_dir, "model.pth"))
        else:
            print("Saving %s/state_dict.pth" % save_dir)
            best_model_state = copy.deepcopy(self.state_dict())
            torch.save(best_model_state, os.path.join(save_dir, "state_dict.pth"))

        # save the config in a yaml file.
        with open(os.path.join(save_dir, "config.yaml"), 'w') as cfg_file:
            try:
                # kinda deprecated by now, but i'll keep it for the old classes.
                yaml.dump(self.config._as_dict(), cfg_file, default_flow_style=False)  # noqa
            except AttributeError:
                if type(self.config) is not dict:
                    yaml.dump(self.config._asdict(), cfg_file, default_flow_style=False)  # noqa
                else:
                    yaml.dump(self.config, cfg_file, default_flow_style=False)  # noqa

    def fit(self, train_data: DataLoader, **kwargs):
        return self.train_fn(
            self,
            train_data,
            **kwargs
        )

    def _update_params_in_config(self, new_config) -> dict:
        r"""

        :param new_config:
        :return:
        """
        # get the keys and make sure they are in alphabetic order. If the were not this could lead to wrong initilizations
        # of layers.
        sorted_keys = sorted(new_config.keys())

        new_dict = dict()
        for _, key in enumerate(self._REQUIRED_PARAMS):
            if key in new_config.keys():
                new_dict[key] = new_config[key]
                continue
            vals = list(new_config[k] for k, in zip(sorted_keys) if key in k.lower())
            if not vals:
                continue
            elif len(vals) == 1:  # this is not pretty but a simple hack for now
                new_dict[key] = vals[0]
            else:
                new_dict[key] = vals

        default_config = self._use_default_config()
        default_config.update(new_dict)
        return default_config

    def _check_params_in_config(self) -> None:
        r"""
        Checks if a required parameter to build the network is not found in the user supplied config.

        Throws:
            ValueError in case a required parameter is not found.

        """
        if type(self.config) is not dict:
            try:
                helper = self.config._as_dict()  # noqa
            except AttributeError:
                helper = self.config._asdict()  # noqa

        helper = self.config

        for _, param in enumerate(type(self)._REQUIRED_PARAMS):
            if param not in helper:
                raise ValueError(
                    f"Parameter \"{param}\" not found. Please make sure to set these parameters: "
                    f"{type(self)._REQUIRED_PARAMS}"
                )
