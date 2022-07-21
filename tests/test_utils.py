import unittest

import numpy as np
import pannt.utils.tools as utils


class MyTestCase(unittest.TestCase):

    def test_ConvertFromWandbConfig_when_RequiredParamsMatchesTarget(self):

        test_wandb_cfg = {
            'lin_neurons1': 128,
            'lin_neurons2': 64,
            'lin_neurons3': 32,
            '_wandb': {},
        }

        target_cfg = {
            'lin_neurons': [128, 64, 32],
        }

        required_params = ["lin_neurons"]

        converted = utils.convert_wandb_config(test_wandb_cfg, required_params)
        self.assertDictEqual(target_cfg, converted)

    def test_ConvertFromWandbConfig_when_RequiredParamsDoesNotMatchTarget(self):

        test_wandb_cfg = {
            'lin_neurons1': 128,
            'lin_neurons2': 64,
            'lin_neurons3': 32,
            '_wandb': {},
        }

        target_cfg = {
            'lin_neurons': [128, 64, 32],
        }

        required_params = ["lin_neurons", "dropout", "channels"]

        converted = utils.convert_wandb_config(test_wandb_cfg, required_params)
        self.assertDictEqual(target_cfg, converted)

    def test_ConvertFromWandbConfig_with_ComplexConfig(self):

        test_wandb_cfg = {
            'channels1': 1,
            'kernel_size': 3,
            'lin_neurons1': 128,
            'channels2': 8,
            'lin_neurons3': 32,
            'pooling_kernel': 2,
            'lin_neurons2': 64,
        }

        target_cfg = {
            'lin_neurons': [128, 64, 32],
            'channels': [1, 8],
            'pooling_kernel': 2,
            'kernel_size': 3,
        }

        required_params = ["lin_neurons", "dropout", "channels", "pooling_kernel", "kernel_size"]

        converted = utils.convert_wandb_config(test_wandb_cfg, required_params)
        self.assertDictEqual(target_cfg, converted)

    def test_ConvertFromWandbConfig_when_TestConfigNotOrdered(self):

        test_wandb_cfg = {
            'lin_neurons2': 64,
            'lin_neurons1': 128,
            'lin_neurons3': 32,
            '_wandb': {},
        }

        target_cfg = {
            'lin_neurons': [128, 64, 32],
        }

        required_params = ["lin_neurons"]

        converted = utils.convert_wandb_config(test_wandb_cfg, required_params)
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


if __name__ == '__main__':
    unittest.main()
