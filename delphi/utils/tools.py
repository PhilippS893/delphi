import nibabel as nib
import pandas as pd
import numpy as np
import numpy.matlib
import torch
import torch.nn.functional as F


class ToTensor(object):

    def __call__(self, data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        else:
            return torch.as_tensor(data)


def convert_to_namedtuple(data: dict, var_name: str = 'config'):
    from collections import namedtuple
    return namedtuple(var_name, data.keys())(*data.values())


def convert_wandb_config(cfg, required_params: list) -> dict:
    """

    :param cfg:
    :param required_params:
    :return:
    """
    cfg = cfg._as_dict()    # convert from wandb to dict

    # get the keys and make sure they are in alphabetic order. If the were not this could lead to wrong initilizations
    # of layers.
    sorted_keys = sorted(cfg.keys())

    new_dict = dict()
    for _, key in enumerate(required_params):
        vals = list(cfg[k] for k, in zip(sorted_keys) if key in k.lower())
        if not vals:
            continue
        elif len(vals) == 1:    # this is not pretty but a simple hack for now
            new_dict[key] = vals[0]
        else:
            new_dict[key] = vals

    new_dict.update(cfg)
    return new_dict


def read_config(yaml_file: str):
    """
    reads a user supplied .yaml configuration file.

    :param yaml_file:
    :param wandb:
    :return:
    """
    import yaml
    from collections import namedtuple

    with open(yaml_file, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    if "_wandb" in params:
        del params["_wandb"]

    return params


def compute_accuracy(real: np.ndarray, predicted: np.ndarray):
    """

    :param real:
    :param predicted:
    :return:
    """
    if len(real) != len(predicted):
        raise ValueError()

    accuracy = np.sum(predicted == real) / len(real)
    return accuracy


def occlude_images(images, attributions, mask, fraction=1, get_fdata=False) -> nib.Nifti1Image:
    """

    Args:
        images:
        attributions:
        mask:
        fraction:   voxel percentage to occlude (default=1%)
        get_fdata:  in case one wants to get the numpy arrays already.

    Returns:

    """
    from nilearn.masking import apply_mask, unmask

    fraction = fraction/100

    # mask the images such that we only really consider brain voxels
    images_masked = apply_mask(images, mask)
    attributions_masked = apply_mask(attributions, mask)

    # find the indices of the largest n voxels for each attribution image (number depends on fraction)
    occlusion_indices = np.argsort(a=attributions_masked, axis=1)[:, ::-1]  # this means we find the largest.
    occlusion_indices = occlusion_indices[:, :int(fraction * occlusion_indices.shape[1])]

    # dummy for occluded images
    occluded_images = np.array(images_masked)
    occluded_images[:, occlusion_indices] = 0

    # convert the occluded_images back into 3d nibabel format
    if not get_fdata:
        return unmask(occluded_images, mask)
    else:
        return unmask(occluded_images, mask).get_fdata()



def get_cnn_output_dim(input_dims, kernel_size, padding=1, stride=1) -> np.ndarray:
    """
    computes the output dimensions after a convolution layer operation.
    Formular: (dims - kernel_size + 2 * padding) / stride + 1
    For reference see: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

    :param input_dims:  dimensions of the input
    :param kernel_size: int describing the kernel size (i.e., the square/cube moving)
    :param padding:     int describing the padding (i.e., the amount of 0s padded to the edges of the input)
    :param stride:      int describing the stride (i.e., stepsize)
    :return:            np.array containing the output dimensions
    """

    if not isinstance(kernel_size, int):
        if len(kernel_size) > 1:
            raise NotImplementedError

    # get number of input dimension
    if type(input_dims) is not np.ndarray:
        input_dims = np.array(input_dims)

    n_dims = input_dims.shape

    kernel_size, padding, stride = np.repeat(kernel_size, n_dims), np.repeat(padding, n_dims), np.repeat(stride, n_dims)

    output_dims = (input_dims - kernel_size + 2 * padding) / stride + 1

    return output_dims.astype(int)


def get_maxpool_output_dim(input_dims, kernel_size, padding, stride, dilation) -> np.ndarray:
    """
    computes the output dimensions after a maxpooling operation.
    Formula: (dims+2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    For reference see: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html

    :param input_dims:  dimensions of the input
    :param kernel_size: int describing the kernel size (i.e., the square/cube moving)
    :param padding:     int describing the padding (i.e., the amount of 0s padded to the edges of the input)
    :param stride:      int describing the stride (i.e., stepsize)
    :param dilation:    int describing the dilation (i.e., space between pooling elements)
    :return:            np.array containing the output dimensions
    """

    if not isinstance(kernel_size, int):
        if len(kernel_size) > 1:
            raise NotImplementedError

    # get number of input dimension
    if type(input_dims) is not np.ndarray:
        input_dims = np.array(input_dims)

    n_dims = input_dims.shape

    kernel_size, padding, stride, dilation = np.repeat(kernel_size, n_dims), np.repeat(padding, n_dims), \
                                             np.repeat(stride, n_dims), np.repeat(dilation, n_dims)

    output_dims = (input_dims + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

    return np.floor(output_dims).astype(int)


def z_transform_volume(volume: np.ndarray) -> np.ndarray:
    """
    z transforms the values within a volume.

    :param volume:  e.g., single fmri volume
    :return:        z transformed volume
    """
    mu = volume.mean()
    std = volume.std()

    return (volume - mu) / std


def z_transform(data: np.ndarray) -> np.ndarray:
    """
    z transforms data along the time axis.

    :param data:    data to be transformed. uses last dimension as "time/sequence"
    :return:        z-transformed data
    """
    mu_ds = data.mean(axis=3)
    sd_ds = data.std(axis=3)

    ds_z = np.empty(data.shape)
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(data.shape[-1]):
            ds_z[:, :, :, i] = (data[:, :, :, i] - mu_ds) / sd_ds

    return ds_z


def save_in_mni(data: np.ndarray, output_name: str):
    """
    Save a 3-D volume in mni space.

    :param output_name: path and name of the file
    :param data: 3d volumetric data
    """
    from delphi import mni_template
    # we want to save the LRP map in MNI space. To do so in an easy (maybe sloppy)
    # way I load the MNI brain mask as a template file and use its header to save
    # the LRP maps.
    template = nib.load(mni_template)

    # save the LRP maps (with the template header) for a given class
    out_data = nib.Nifti1Image(data, template.affine, header=template.header)
    print('Saving %s' % output_name)
    nib.save(out_data, output_name)


def classify_volumes(data, net):
    """

    :param data:
    :param net:
    :return:
    """
    net.eval()

    with torch.no_grad():
        x = torch.tensor(np.nan_to_num(data)).cpu()
        # add the two necessary dimensions for pytorch (batch_size, n_channels)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)

        outputs = F.softmax(net(x.float()), dim=1)
        out = outputs.data.cpu()

    return out


def compute_stats(classify_tc, ev_files, n_vols, tr=.72):
    """

    :param classify_tc:
    :param ev_files:
    :param n_vols:
    :param tr:
    :return:
    """
    n_evs = len(ev_files)
    chance_thresh = 1 / n_evs
    r_evtc_with_bin = np.zeros(n_evs)
    counts = np.empty((6, n_evs))

    for ev in range(n_evs):
        event_tc = np.zeros(n_vols)
        ev_info = pd.read_table(ev_files[ev], header=None)
        # print(ev_info)
        ev_start = np.array(np.ceil(ev_info[0] / tr)).astype(int)
        ev_end = np.array(ev_start + np.ceil(ev_info[1] / tr)).astype(int)

        for i in range(len(ev_start)):
            if 'rest' in ev_files[ev]:
                indices = np.arange(ev_start[i], ev_end[i])
            else:
                indices = np.arange(ev_start[i], ev_end[i]) + 4  # account for the hemodynamic delay
            event_tc[indices] = 1

        counts[0, ev] = np.sum(classify_tc[event_tc == 1, ev] > chance_thresh)
        counts[1, ev] = np.sum(classify_tc[event_tc == 0, ev] > chance_thresh)
        counts[2, ev] = np.sum(classify_tc[event_tc == 0, ev] < chance_thresh)
        counts[3, ev] = np.sum(classify_tc[event_tc == 1, ev] < chance_thresh)
        counts[4, ev] = len(classify_tc[event_tc == 1, ev])
        counts[5, ev] = len(classify_tc[event_tc == 0, ev])

        # binarize the classification vector (x > .2 = 1 and < .2 = 0)
        bin_classification = np.zeros(n_vols)
        bin_classification[classify_tc[:, ev] > chance_thresh] = 1

        # compute the correlation between the event timing and the binarized
        # classification confidence
        r_evtc_with_bin[ev] = np.corrcoef(event_tc, bin_classification)[0, 1]

    stats = {
        'tp': counts[0, :],  # true positives
        'fp': counts[1, :],  # false positives
        'tn': counts[2, :],  # true negatives
        'fn': counts[3, :],  # false negatives
        'nPos': counts[4, :],  # number of actual positives
        'nNeg': counts[5, :],  # number of actual negatives
        'r_with_model': r_evtc_with_bin  # correlation of binarized prediction with event timecourse
    }

    return stats


def combine_dict(d1, d2, dim=0):
    """

    :param dim:
    :param d1:
    :param d2:
    :return:
    """

    if not d1:
        d1 = d2.copy()
    else:
        for key in d1.keys():
            if dim == 0:
                d1[key] = np.vstack((d1[key], d2[key]))
            elif dim == 1:
                d1[key] = np.hstack((d1[key], d2[key]))
            else:
                print('Wrong dimension to stack')

    return d1


def running_stats(prev_mean, prev_dsqr, x, n):
    """
    computes the running mean etc. Important function to normalize incoming data in real-time.

    :param prev_mean:
    :param prev_dsqr:
    :param x:
    :param n:
    :return:
    """
    new_mean = prev_mean + (x - prev_mean) / n
    new_dsqr = prev_dsqr + ((x - new_mean) * (x - prev_mean))

    if n <= 2:
        std = np.sqrt(new_dsqr / n)
    else:
        std = np.sqrt(new_dsqr / (n - 1))

    return new_mean, new_dsqr, std


def to_pd_dataframe(data, labels, datatype, smooth_factor):
    """

    :param smooth_factor:
    :param datatype:
    :param data:
    :param labels:
    :return:
    """

    # check if labels is a numpy array
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # 1. flatten the matrices
    for c, key in enumerate(data.keys()):
        if c == 0:
            n_entries = data[key].shape[0]
        data[key] = data[key].flatten('F')

    # 2. add the labels
    data['label'] = np.matlib.repmat(labels, n_entries, 1).flatten('F')

    # 3. convert to dataframe
    df = pd.DataFrame.from_dict(data)

    # 4. add some extra columns
    df['datatype'] = datatype
    df['smooth_factor'] = smooth_factor
    df['recall'] = df.tp / df.nPos
    df['specificity'] = df.tn / df.nNeg
    df['precision'] = df.tp / (df.tp + df.fp)
    df['accuracy'] = (df.tp + df.tn) / (df.nPos + df.nNeg)

    return df
