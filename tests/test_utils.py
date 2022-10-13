import unittest
import os
import wandb
import numpy as np
import delphi.utils.tools as utils
from delphi.networks.ConvNets import BrainStateClassifier3d

os.environ['WANDB_MODE'] = 'offline'


class MyTestCase(unittest.TestCase):

    def test_myshit(self):
        test_wandb_cfg = {
            'channels1': 1,
            'channels2': 8,
            'channels3': 16,
            'lin_neurons1': 256,
            'lin_neurons2': 128,
            'batch_size': 32,
        }

        wandb.init(config=test_wandb_cfg, entity='philis893', project='code-testing',
                   dir='')
        target_cfg = {
            'lin_neurons': [128, 64, 32],
            'batch_size': 32,
        }
        converted = utils.convert_wandb_config(wandb.config, BrainStateClassifier3d._REQUIRED_PARAMS)
        self.assertDictEqual(target_cfg, converted)

    def test_ConvertFromWandbConfig_when_RequiredParamsMatchesTarget(self):
        test_wandb_cfg = {
            'lin_neurons1': 128,
            'lin_neurons2': 64,
            'lin_neurons3': 32,
            'batch_size': 32,
        }

        wandb.init(config=test_wandb_cfg, entity='ml4ni', project='code-testing',
                   dir='')

        target_cfg = {
            'lin_neurons': [128, 64, 32],
            'batch_size': 32,
        }

        required_params = ["lin_neurons"]

        converted = utils.convert_wandb_config(wandb.config, required_params)
        self.assertDictEqual(target_cfg, converted)

    def test_ConvertFromWandbConfig_when_RequiredParamsDoesNotMatchTarget(self):
        test_wandb_cfg = {
            'lin_neurons1': 128,
            'lin_neurons2': 64,
            'lin_neurons3': 32,
        }

        wandb.init(config=test_wandb_cfg, entity='ml4ni', project='code-testing',
                   dir='')

        target_cfg = {
            'lin_neurons': [128, 64, 32],
        }

        required_params = ["lin_neurons", "dropout", "channels"]

        converted = utils.convert_wandb_config(wandb.config, required_params)
        self.assertDictEqual(target_cfg, converted)

    def test_ConvertFromWandbConfig_when_TestConfigNotOrdered(self):
        test_wandb_cfg = {
            'lin_neurons2': 64,
            'lin_neurons1': 128,
            'lin_neurons3': 32,
        }

        wandb.init(config=test_wandb_cfg, entity='ml4ni', project='code-testing',
                   dir='')

        target_cfg = {
            'lin_neurons': [128, 64, 32],
        }

        required_params = ["lin_neurons"]

        converted = utils.convert_wandb_config(wandb.config, required_params)
        self.assertDictEqual(target_cfg, converted)

    def test_calculate_accuracy(self):
        real = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        predicted = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

        acc = utils.compute_accuracy(real, predicted)
        self.assertEqual(acc, .6)

    def test_ToTensor_from_ndarray(self):
        import torch
        func = utils.ToTensor()
        tensor = func(np.array([1, 2, 3, 4]))
        self.assertIsInstance(tensor, torch.Tensor)

    def test_ToTensor_from_list(self):
        import torch
        func = utils.ToTensor()
        tensor = func([1, 2, 3, 4])
        self.assertIsInstance(tensor, torch.Tensor)

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree("wandb", ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
