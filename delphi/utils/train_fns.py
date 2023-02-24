import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Tuple
from tqdm.auto import tqdm


def standard_train(
        model,
        train_data: DataLoader,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam, lr: float = .00001,
        device: torch.device = torch.device("cpu"),
        train=True,
        **optimizer_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    A simple function to train a supplied neural network for multiclass problems.

    Args:
        model: only necessary if this function is not supplied to the model constructor. Otherwise model.fit()
            calls the function with model=self
        train_data (DataLoader): dataloader used for training or validation/test if function is called in model.eval()
            context
        loss_fn: loss function to use (default: CrossEntropyLoss)
        optimizer: optimizer to use to adjust weights in backward pass (default: Adam)
        lr (float): the learning rate for the optimizer
        device (torch.device): do computations on device, e.g., cpu or gpu (default: cpu)
        train (bool): set the network into training mode (True|default) or evaluation (False)
        **optimizer_kwargs: additional arguments for the supplied optimizer

    Returns:
        Tuple[epoch_loss, stats]: the stats variable contains multiple values. The first 0:n_classes columns contains
        the classification probability for a given input (row). The second to last column (i.e., stats[:,-2]) contains
        the predicted label. The last column (i.e. stats[:, -1] contains the real label.

    """

    # set the model into training mode if it is not and train=True
    if train:
        if not model.training:
            model.train()
    else:
        model.eval()

    # here is something new: the optimizer
    # The optimizer determines the algorithm with which the weights of the layers
    # are adjusted. Here we use the 'Adam' algorithm by default.
    optimizer = optimizer(model.parameters(), lr=lr, **optimizer_kwargs)

    epoch_loss = 0

    # the batch loop. Within this loop we iterate over all samples stored in the
    # train_data variable.
    # the variable 'batch' represents the current iteration [integer value]
    # the variable 'inputs' is the actual input data in the shape of batch_size-by-inputshape
    # the variable 'labels' contains the respective class label
    for batch, (inputs, labels) in tqdm(enumerate(train_data)):

        # in this function we transfer the data and the label tensors to the chosen device
        # NOTE: I RECOMMEND TRANSFERING THE DATA AND LABELS TO THE DEVICE WHEN YOU LOAD THEM
        # MORE ON THIS IN A DIFFERENT STEP THOUGH [see DataSets and DataLoaders)
        #inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs, labels

        # reset the gradients
        # a crucial step in training the networks. Otherwise the gradients accumulate after
        # each batch iteration and weird stuff will happen.
        if model.training:
            for p in model.parameters():
                p.grad = None

        outputs = model(inputs.float())  # forward pass through model

        loss = loss_fn(outputs.squeeze(), labels.squeeze())  # calculate loss

        # in this statement we check if the network is currently in training mode,
        # which means, that every layer in the network has the required_grad flag set
        # to True. This in turn means that the backpropagation algorithm is executed.
        # If the model, however, is not in training mode we do not want to exectue
        # the backward pass and we also do not want to store the so-called pytorch graph.
        if model.training:
            loss.backward()  # do a backward pass
            optimizer.step()  # update parameters

        # get the probabilities of the predictions
        prediction_probs = model.SM(outputs.data).cpu().numpy()
        # get the label number of the output
        _, predicted_labels = torch.max(outputs.data, 1)

        epoch_loss += loss.item()  # sum up the loss over all batches

        # just a helper variable
        inter = np.hstack(
            [prediction_probs.squeeze(),
             labels.squeeze().cpu().numpy()[:, None],
             predicted_labels.squeeze().cpu().numpy()[:, None]]
        )
        stats = np.vstack([stats, inter]) if 'stats' in vars() else inter  # noqa

    return epoch_loss / len(train_data), stats  # noqa
