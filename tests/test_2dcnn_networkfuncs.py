import unittest
import torch
import pathlib as pl
from pannt.networks.ConvNets import Simple2dCnnClassifier
from pannt.utils.train_fns import standard_train


def test_train_fn(model, data):
    return True


class TestNetworkFunctions(unittest.TestCase):
    cnn_target_config = {
        'channels': [1, 8, 16, 32],
        'kernel_size': [5, 5, 5],
        'pooling_kernel': 2,
        'lin_neurons': [256, 128],
        'dropout': .5,
        'n_classes': 10,
        'input_dims': (28, 28),
        'train_fn': standard_train,
        'last_cnn_dims': [3, 3],
    }

    cnn_test_model_name = 'generated_test_files/cnn2d_test_model_name'
    cnn_test_model_save_with_torch = 'generated_test_files/cnn2d_saved_with_torch.pth'
    cnn_test_model_with_custom_train = 'generated_test_files/cnn2d_model_with_custom_train.pth'

    def test_network_default_config_dict(self):
        r"""
        Test if the default config of the network is set as expected.

        :return:
        """
        model = Simple2dCnnClassifier((28, 28), 10)
        self.assertDictEqual(self.cnn_target_config, model.config)

    def test_network_config_from_wandb(self):
        r"""
        Test if the config of the network is set as expected when using the weights&biases config.

        :return:
        """
        import wandb
        import os
        from pannt.utils.tools import read_config

        test_params = read_config('cnn_test_params.yaml')

        # these are the changes we expect
        target_config = {
            'kernel_size': [3, 3, 3, 3],
            'input_dims': (28, 28),
            'channels': [1, 8, 16, 32, 64],
            'dropout': .7,
            'pooling_kernel': 2,
            'train_fn': test_train_fn,
            'lin_neurons': [128, 64],
            'n_classes': 10,
            'last_cnn_dims': [1, 1],
            '_wandb': {},
        }

        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(config=test_params, entity='ml4ni', project='code-testing',
                   dir='')
        config = wandb.config
        model = Simple2dCnnClassifier((28, 28), 10, config._as_dict(), test_train_fn)
        wandb.finish()
        self.assertDictEqual(target_config, model.config)

    def test_custom_config(self):
        r"""

        :return:
        """
        test_config = {
            'lin_neurons': [32, 16]
        }

        target_config = self.cnn_target_config.copy()
        target_config['lin_neurons'] = [32, 16]

        model = Simple2dCnnClassifier((28, 28), 10, test_config)
        self.assertDictEqual(target_config, model.config)

    def test_for_custom_save_dir(self):
        r"""
        Test if the user supplied directory is created. This is the umbrella test for the model.save() function. It has
        two "child" tests that check if the state_dict.pth and config.yaml files are created.

        :return:
        """
        target_dir = pl.Path(self.cnn_test_model_name)
        model = Simple2dCnnClassifier((28, 28), 10)
        model.save(str(target_dir))
        self.assertEqual((str(target_dir), target_dir.is_dir()), (str(target_dir), True))

    def test_for_custom_save_config(self):
        r"""
        Test if the config.yaml file is created when using model.save()
        :return:
        """
        target_dir = pl.Path(self.cnn_test_model_name)
        target_config_file = pl.Path(target_dir, 'config.yaml')
        model = Simple2dCnnClassifier((28, 28), 10)
        model.save(str(target_dir))
        self.assertEqual((str(target_config_file), target_config_file.is_file()), (str(target_config_file), True))

    def test_for_custom_save_model(self):
        r"""
        Test if the state_dict.pth file is created when using model.save()
        :return:
        """
        target_dir = pl.Path(self.cnn_test_model_name)
        target_model_file = pl.Path(target_dir, 'state_dict.pth')
        model = Simple2dCnnClassifier((28, 28), 10)
        model.save(str(target_dir))
        self.assertEqual((str(target_model_file), target_model_file.is_file()), (str(target_model_file), True))

    def test_save_with_custom_train_fn(self):
        r"""

        :return:
        """
        model = Simple2dCnnClassifier((28, 28), 10, test_train_fn)
        save_path = pl.Path(self.cnn_test_model_with_custom_train)
        torch.save(model, save_path)
        self.assertEqual((str(save_path), save_path.is_file()), (str(save_path), True))

    def test_save_model_with_torch(self):
        r"""
        Test if the torch.save() method works to save the entire model.

        :return:
        """
        save_path = pl.Path(self.cnn_test_model_save_with_torch)
        model = Simple2dCnnClassifier((28, 28), 10)
        torch.save(model, self.cnn_test_model_save_with_torch)
        self.assertEqual((str(save_path), save_path.is_file()), (str(save_path), True))

    def test_load_model_with_torch(self):
        r"""
        Test if the torch.load() method works to load the entire model.

        :return:
        """
        model = torch.load(self.cnn_test_model_save_with_torch)
        self.assertDictEqual(self.cnn_target_config, model.config)

    def test_custom_train_fn(self):
        r"""

        :return:
        """
        from torch.utils.data import DataLoader, Dataset
        model = Simple2dCnnClassifier((28, 28), 10, test_train_fn)

        self.assertEqual(True, model.fit(DataLoader(Dataset())))

    def test_network_config_from_load_pretrained(self):
        r"""
        Test if the network config set by the default is still valid when loading from save state.

        :return:
        """
        model = Simple2dCnnClassifier(self.cnn_test_model_name)
        self.assertDictEqual(self.cnn_target_config, model.config)

    def test_load_with_custom_train_fn(self):
        r"""

        :return:
        """
        target_config = self.cnn_target_config.copy()
        target_config['train_fn'] = test_train_fn
        model = torch.load(self.cnn_test_model_with_custom_train)
        self.assertDictEqual(target_config, model.config)

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree("wandb", ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
