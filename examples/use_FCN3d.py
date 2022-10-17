from delphi.networks.ConvNets import FCN3d
from delphi.utils.datasets import NiftiDataset
from torch.utils.data import DataLoader
import delphi.utils as utils
import os, torch
import numpy as np
from glob import glob
from torchinfo import summary
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Gamma, Pass
from zennit.types import Convolution, Linear, Activation
from zennit.canonizers import SequentialMergeBatchNorm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg = {
    "n_hidden_layers": 3,
    "kernel_size": [3] * 3,
    "add_batchnorm": [True] * 3,
    "add_pooling": [True] * 3,
    "conv_kwargs": {
    }
}

class_labels = sorted(["footleft", "handleft", "footright", "tongue", "handright"])

model = FCN3d((91, 109, 91), len(class_labels), cfg)

print(summary(model, (1, 1, 91, 109, 91)))

data_dir = "/Users/phisei/Documents/phd/thesis_code_and_analyses/t-maps"

dl_train = DataLoader(
    NiftiDataset(
        os.path.join(data_dir, "train"),
        class_labels,
        0,
        device=DEVICE,
        transform=utils.tools.ToTensor()
    ),
    batch_size=4,
    shuffle=True
)

dl_test = DataLoader(
    NiftiDataset(
        os.path.join(data_dir, "test"),
        class_labels,
        0,
        device=DEVICE,
        transform=utils.tools.ToTensor()
    ),
    batch_size=20,
    shuffle=False
)

best_loss, best_acc = 100, 0
patience, patience_ctr = 9, 0
overwrite = True
if os.path.exists("example_models/example_fcn3d") and overwrite:
    for epoch in range(100):
        _, _ = model.fit(dl_train, lr=.00001, device=DEVICE)

        with torch.no_grad():
            tloss, tstats = model.fit(dl_train, train=False)
            testloss, teststats = model.fit(dl_test, train=False)

        tacc = utils.tools.compute_accuracy(tstats[:, -2], tstats[:, -1])
        testacc = utils.tools.compute_accuracy(teststats[:, -2], teststats[:, -1])

        if testloss < best_loss and testacc > best_acc:
            best_loss, best_acc = testloss, testacc
            model.save("example_models/example_fcn3d")
            patience_ctr = 0
        else:
            patience_ctr += 1

        print(f"Epoch{epoch}, train-loss = {tloss}, train-acc = {tacc}, test-loss = {testloss}, test-acc = {testacc}")

        if patience_ctr > patience:
            print("Performance did not improve for 10 epochs. Stopping early")
            break
else:
    print("Model already exists.")

model = FCN3d((91, 109, 91), 5, cfg)
model.load_state_dict(torch.load("example_models/example_fcn3d/state_dict.pth"))
model.eval()

# setup LRP attribution
composite_lrp_map = [
    (Activation, Pass()),
    (Convolution, Gamma(gamma=0.25)),
    # according to: https://link-springer-com.stanford.idm.oclc.org/chapter/10.1007/978-3-030-28954-6_10
    (Linear, Epsilon(epsilon=0))  # epsilon=0 -> LRP-0 rule
]
canonizer = SequentialMergeBatchNorm()
LRP = LayerMapComposite(
    layer_map=composite_lrp_map,
    canonizers=[canonizer]
)

lrp_out_dir = "example_models/example_fcn3d/lrp-maps"
for l, label in enumerate(class_labels):

    dl = DataLoader(
        NiftiDataset(
            os.path.join(data_dir, "test"),
            [label],
            0,
            device=DEVICE,
            transform=utils.tools.ToTensor()
        ),
        batch_size=20,
        shuffle=False
    )

    if not os.path.exists(lrp_out_dir):
        os.mkdir(lrp_out_dir)

    for i, (image, target) in enumerate(dl):
        image.requires_grad = True
        grad_dummy = torch.eye(len(class_labels))[[target]]

        with LRP.context(model) as modified_model:
            output = modified_model(image)
            attribution, = torch.autograd.grad(
                output,
                image,
                grad_outputs=grad_dummy
            )

            indi_lrp = np.moveaxis(attribution.squeeze().detach().numpy(), 0, -1)
            indi_lrp = utils.tools.z_transform_volume(indi_lrp)

    utils.tools.save_in_mni(indi_lrp, os.path.join(lrp_out_dir, '%s.nii.gz' % label))
