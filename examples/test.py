import delphi.networks._utils.hooks as net_hooks
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch
import numpy as np
from delphi.explain.cka import *
import matplotlib.pyplot as plt


test_data = MNIST('../../dl4ni/notebooks/data/', train=False, download=False, transform=ToTensor())

model_cnn = torch.load("../../dl4ni/notebooks/my_cnn_mnist/model.pth")
model_cnn.eval()
net_hooks.hook_net(model_cnn)

model_cnn2 = torch.load("../../dl4ni/notebooks/my_cnn2_mnist/model.pth")
model_cnn2.eval()
net_hooks.hook_net(model_cnn2)

activs_a, activs_b = {}, {}

for i in test_data.targets.unique():
    idx = np.where(test_data.targets == i)
    dl = DataLoader(Subset(test_data, idx[0][:200]), batch_size=5)

    for batch, (img, lbl) in enumerate(dl):
        a = model_cnn(img.float())
        activs_a = net_hooks.assign_activations(activs_a)
        a = model_cnn2(img.float())
        activs_b = net_hooks.assign_activations(activs_b)

r = np.corrcoef(np.reshape(activs_a['convlayer0'][:, 0, :, :], (len(activs_a['convlayer0']), np.prod(activs_a['convlayer0'].shape[-2:]))))

cka_mat = {}
fig, axes = plt.subplots(1, 3)
for i, key in enumerate(activs_a.keys()):
    if "conv" in key:
        cka_mat[key] = get_within_layer_cka_matrix(activs_a[key])
        axes[i].imshow(cka_mat[key])
cka_mat = get_cka_matrix(activs_a, activs_b)