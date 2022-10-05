import nibabel as nib  # nibabel is for loading nifti files
import glob
from torch.utils.data import Dataset
import os
import torch
import pandas as pd
import numpy as np
from typing import List
from PIL import Image


# we call this class TabularDataset since that is what it is
class TabularDataset(Dataset):
    r"""
    Class to take care of tabular files such as .csv, .xls, etc.
    """

    POSSIBLE_EXCEL_EXTENSIONS = [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]
    EXPECTED_LABEL_COLUMN_NAMES = ["class", "label", "target"]

    ####### REQUIRED CLASS FUNCTIONS ########

    # now we come to the so-called constructor or the initializer function
    # I personally like having the option to add transformation functions and
    # the option to shuffle the labels. This allows me to quickly create
    # a null distribution/performance estimate
    def __init__(self, path_to_file, device=torch.device("cpu"), shuffle_labels=False, transform=None):
        r"""
        The constructor of the TabularDataset class.

        Args:
            path_to_file (str): the path to the file. Supports .csv, .xls file-types at the moment
            device (torch.device): the device on which to store the data
            shuffle_labels (bool): Default=False; permutes the class labels
            transform: can be a list of functions to transform the data
        """
        super(TabularDataset, self).__init__()

        self.path_to_file = path_to_file
        self.shuffle_labels = shuffle_labels
        self.transform = transform
        self.device = device

        # we can check what the file extension of the supplied file is.
        # this informs us which function to use to read the file.
        filename, file_extension = os.path.splitext(self.path_to_file)

        # read the file in path_to_file with pandas reading functions
        if file_extension == '.csv':
            self.data = pd.read_csv(self.path_to_file)

        elif file_extension in self.POSSIBLE_EXCEL_EXTENSIONS:
            self.data = pd.read_excel(self.path_to_file)

        else:
            raise ValueError(f"{file_extension} is not \'.csv\' or one of {self.POSSIBLE_EXCEL_EXTENSIONS}")

        self.label_column = self._check_for_label_column()

    def __len__(self):
        r"""
        returns the length, i.e. the number of samples, of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        r"""
        """
        data = self.data.loc[:, self.data.columns != self.label_column].to_numpy()
        sample = data[idx, :]
        label = self.data[self.label_column][idx]

        # In case you provide a set of transformations execute them here
        if self.transform:
            label = self.transform(label).to(self.device)
            sample = self.transform(sample).float().to(self.device)

        return (sample, label)

    ####### CUSTOM / HELPER FUNCTIONS ########

    def _check_for_label_column(self):
        r"""
        make sure the dataset has a column indicating the label, class, or target

        Returns:
            label_column: the column name of the target/class/label
        """
        # make sure all column values are lowercase
        columns = [column_name.lower() for column_name in self.data.columns.to_list()]

        #
        label_column = [col_name for col_name in self.EXPECTED_LABEL_COLUMN_NAMES if columns.count(col_name) > 0]

        if not label_column:
            raise ValueError(f"Did not find a column indicating the {self.EXPECTED_LABEL_COLUMN_NAMES}")

        return label_column[0]


class ImageDataset(Dataset):
    r"""
    Dataset class to load 2d images.
    """

    _SUPPORTED_IMG_FTYPES = ['jpg', 'jpeg', 'png']

    def __init__(
            self,
            data_dir: str,
            labels: List[str],
            n_samples: int = 0,
            img_ftype: str = "jpg",
            device: torch.device = torch.device("cpu"),
            shuffle_labels: bool = False,
            transform=None
    ) -> None:
        """
        :param data_dir:
        :param labels:
        :param device:
        :param transform:
        """
        super(ImageDataset, self).__init__()
        self.data_dir = data_dir
        self.classes = labels
        self.n_samples = n_samples
        self.img_ftype = img_ftype
        self.device = device
        self.shuffle_labels = shuffle_labels
        self.transform = transform

        if img_ftype not in self._SUPPORTED_IMG_FTYPES:
            raise ValueError(f"Supplied image file type {img_ftype} not in {self._SUPPORTED_IMG_FTYPES}")

        # get the file paths and labels
        for iLabel in range(len(labels)):
            # look for all files in alphanumerical order in the label directory
            file_names = sorted(glob.glob(os.path.join(data_dir, labels[iLabel], f"*.{img_ftype}")))
            # select only the requested number of files if n > 0
            n_files = len(file_names[:n_samples]) if n_samples != 0 else len(file_names)

            if iLabel == 0:
                self.data = np.array(file_names[:n_files])
                self.labels = np.array(np.repeat(labels[iLabel], n_files))
            else:
                self.data = np.append(self.data, file_names[:n_files])
                self.labels = np.append(self.labels, np.repeat(labels[iLabel], n_files))

        if shuffle_labels:
            self.labels = np.random.permutation(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        """

        :param idx:
        :return:
        """
        img = Image.open(self.data[idx])

        label = np.squeeze(np.where(np.array(self.labels[idx]) == np.array(self.classes)))

        # TODO: this is not great. Needs adjustment. Concerns all other Dataset classes!
        if self.transform:
            label = torch.tensor(label).to(self.device)
            img = self.transform(img).to(self.device)

        return img, label


class NiftiDataset(Dataset):
    """
      NiftiLoader has torch functionality to rapidly generate and load new
      batches for training and testing.
    """

    def __init__(self, data_dir, labels, n, device, dims=3, shuffle_labels=False, transform=None):
        """
        Constructor for the NiftiDataset class

        :param data_dir:        path to the data
        :param labels:          list of class names (directories within data_dir)
        :param n:               the number of samples to load. If "0" take every example in directory.
        :param device:          the device to use (cpu|gpu)
        :param dims:            3 to keep the dimension, 1 to flatten into vector
        :param shuffle_labels:  in case one wants to train a null-model enable label shuffling. Using this for training
                                should lead to a network that provides information if labels would not matter. I.e.,
                                it should perform only at chance level.
        :param transform:       A composition of transformation functions that should be applied to the data.
        """

        self.device = device
        self.classes = labels
        self.dims = dims
        self.transform = transform

        # get the file paths and labels
        for iLabel in range(len(labels)):
            # look for all files in alphanumerical order in the label directory
            file_names = sorted(glob.glob(os.path.join(data_dir, labels[iLabel], "*.nii.gz")))
            # select only the requested number of files if n > 0
            n_files = len(file_names[:n]) if n != 0 else len(file_names)

            if iLabel == 0:
                self.data = np.array(file_names[:n_files])
                self.labels = np.array(np.repeat(labels[iLabel], n_files))
            else:
                self.data = np.append(self.data, file_names[:n_files])
                self.labels = np.append(self.labels, np.repeat(labels[iLabel], n_files))

        if shuffle_labels:
            self.labels = np.random.permutation(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        load a (batch) sample. This is usually done automatically by the Pytorch DataLoader class.

        :param idx: the index of the sample to load
        :return: tuple(volume, label)
        """

        # make sure that there are no NaNs in the data.
        volume = np.nan_to_num(nib.load(self.data[idx]).get_fdata())

        volume[np.isnan(volume)] = 0  # this one is in here because I am paranoid

        # sometimes nibabel retains the temporal dimension. (x, y, z, t)
        # we do not want that so we get rid of it.
        if len(volume.shape) > 3:
            volume = volume.squeeze()

        volume = np.expand_dims(volume, 0) if self.dims == 3 else volume.flatten()  # add the channel dimension
        label = np.squeeze(np.where(np.array(self.labels[idx]) == np.array(self.classes)))

        # In case you provide a set of transformations execute them here
        if self.transform:
            label = self.transform(label).to(self.device)
            volume = self.transform(volume).float().to(self.device)
        else:
            label = label.to(self.device)
            volume = volume.to(self.device)

        return volume, label
