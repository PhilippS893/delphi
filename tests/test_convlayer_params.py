import unittest
import numpy as np
from delphi.utils.tools import get_cnn_output_dim, get_maxpool_output_dim


class TestLayerDimFunctions(unittest.TestCase):

    def test_cnn_output_dim_2d_kernel(self):
        out_dims = get_cnn_output_dim([10, 10], 3, padding=0, stride=1)
        np.testing.assert_array_equal(out_dims, np.array([8, 8]))

    def test_cnn_output_dim_2d_kernel_rectangle(self):
        out_dims = get_cnn_output_dim([10, 10], 3, padding=0, stride=1)
        np.testing.assert_array_equal(out_dims, np.array([8, 8]))

    def test_cnn_output_dim_2d_padding(self):
        self.assertRaises(NotImplementedError, get_cnn_output_dim, [10, 10], [3, 5], 1, 1)

    def test_cnn_output_dim_2d_padding_stride(self):
        out_dims = get_cnn_output_dim([10, 10], 3, padding=1, stride=2)
        np.testing.assert_array_equal(out_dims, np.array([5, 5]))

    def test_maxpool_output_dim_2d_kernel(self):
        out_dims = get_maxpool_output_dim([10, 10], 2, padding=0, stride=2, dilation=1)
        np.testing.assert_array_equal(out_dims, np.array([5, 5]))

    def test_maxpool_output_dim_2d_dilation(self):
        out_dims = get_maxpool_output_dim([7, 7], 3, padding=0, stride=1, dilation=2)
        np.testing.assert_array_equal(out_dims, np.array([3, 3]))

    def test_maxpool_output_dim_2d_padding_dilation(self):
        out_dims = get_maxpool_output_dim([7, 7], 3, padding=1, stride=1, dilation=2)
        np.testing.assert_array_equal(out_dims, np.array([5, 5]))


if __name__ == '__main__':
    unittest.main()
