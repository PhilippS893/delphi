# delphi
seiDEL PHIlipp's artificial neural network toolbox

<p align="justify">
I wrote this toolbox in tandem with my dissertation. The goal for this dissertation was to make it easy and quickly
to train and test, primarily, 3D convolutional neural networks (CNNs) with different parameters and potentially different
architectures.
</p> 

## Creating networks

Below are two examples of how to generate a CNN:
```python
from delphi.networks.ConvNets import BrainStateClassifier3d
from torchinfo import summary

model_cfg = {
    "channels": [1, 8, 16, 32, 64],
    "kernel_size": 3,
    "pooling_kernel": 2,
    "lin_neurons": [128, 64],
}

input_dims = (91, 109, 91)
n_classes = 5

# creates a 3D CNN with 4 convolutional layers with 3x3x3 kernel size and
# 8, 16, 32, and 64 filters, respectively.
# It also adds two linear layers with 128 and 64 neurons and hooks it up
# to 5 output neurons.
model = BrainStateClassifier3d(input_dims, n_classes, model_cfg)
print(summary(model, (1, 1, 91, 109, 91)))
#OUTPUT
"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
BrainStateClassifier3d                   [1, 5]                    --
├─Dropout: 1-6                           [1, 128]                  --
├─ReLU: 1-2                              --                        --
├─Softmax: 1-3                           --                        --
├─Sequential: 1-4                        --                        --
│    └─Sequential: 2-1                   [1, 8, 45, 54, 45]        --
│    │    └─Conv3d: 3-1                  [1, 8, 91, 109, 91]       224
│    │    └─ReLU: 3-2                    [1, 8, 91, 109, 91]       --
│    │    └─MaxPool3d: 3-3               [1, 8, 45, 54, 45]        --
│    └─Sequential: 2-2                   [1, 16, 22, 27, 22]       --
│    │    └─Conv3d: 3-4                  [1, 16, 45, 54, 45]       3,472
│    │    └─ReLU: 3-5                    [1, 16, 45, 54, 45]       --
│    │    └─MaxPool3d: 3-6               [1, 16, 22, 27, 22]       --
│    └─Sequential: 2-3                   [1, 32, 11, 13, 11]       --
│    │    └─Conv3d: 3-7                  [1, 32, 22, 27, 22]       13,856
│    │    └─ReLU: 3-8                    [1, 32, 22, 27, 22]       --
│    │    └─MaxPool3d: 3-9               [1, 32, 11, 13, 11]       --
│    └─Sequential: 2-4                   [1, 64, 5, 6, 5]          --
│    │    └─Conv3d: 3-10                 [1, 64, 11, 13, 11]       55,360
│    │    └─ReLU: 3-11                   [1, 64, 11, 13, 11]       --
│    │    └─MaxPool3d: 3-12              [1, 64, 5, 6, 5]          --
├─Dropout: 1-5                           [1, 9600]                 --
├─Sequential: 1                          --                        --
│    └─Sequential: 2-5                   [1, 128]                  --
│    │    └─Linear: 3-13                 [1, 128]                  1,228,928
│    │    └─ReLU: 3-14                   [1, 128]                  --
├─Dropout: 1-6                           [1, 128]                  --
├─Sequential: 1                          --                        --
│    └─Sequential: 2-6                   [1, 64]                   --
│    │    └─Linear: 3-15                 [1, 64]                   8,256
│    │    └─ReLU: 3-16                   [1, 64]                   --
├─Linear: 1-7                            [1, 5]                    325
==========================================================================================
Total params: 1,310,421
Trainable params: 1,310,421
Non-trainable params: 0
Total mult-adds (M): 851.24
==========================================================================================
Input size (MB): 3.61
Forward/backward pass size (MB): 75.92
Params size (MB): 5.24
Estimated Total Size (MB): 84.77
==========================================================================================
"""
```

<p align="justify">A second example shows how the <code>channel</code> variable can be split up into different key-value pairs to further 
configure the network.</p>

```python
from delphi.networks.ConvNets import BrainStateClassifier3d
from torchinfo import summary

model_cfg = {
    "channels1": 1, #the first channel always pertains the input channels. Could also be >1
    "channels2": 16,
    "channels3": 8,
    "channels4": 32,
    "kernel_size": 3,
    "pooling_kernel": 2,
    "lin_neurons1": 64,
    "lin_neurons2": 128,
}

input_dims = (91, 109, 91)
n_classes = 5

model = BrainStateClassifier3d(input_dims, n_classes, model_cfg)
print(summary(model, (1, 1, 91, 109, 91)))
#OUTPUT:
"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
BrainStateClassifier3d                   [1, 5]                    --
├─Dropout: 1-6                           [1, 64]                   --
├─ReLU: 1-2                              --                        --
├─Softmax: 1-3                           --                        --
├─Sequential: 1-4                        --                        --
│    └─Sequential: 2-1                   [1, 16, 45, 54, 45]       --
│    │    └─Conv3d: 3-1                  [1, 16, 91, 109, 91]      448
│    │    └─ReLU: 3-2                    [1, 16, 91, 109, 91]      --
│    │    └─MaxPool3d: 3-3               [1, 16, 45, 54, 45]       --
│    └─Sequential: 2-2                   [1, 8, 22, 27, 22]        --
│    │    └─Conv3d: 3-4                  [1, 8, 45, 54, 45]        3,464
│    │    └─ReLU: 3-5                    [1, 8, 45, 54, 45]        --
│    │    └─MaxPool3d: 3-6               [1, 8, 22, 27, 22]        --
│    └─Sequential: 2-3                   [1, 32, 11, 13, 11]       --
│    │    └─Conv3d: 3-7                  [1, 32, 22, 27, 22]       6,944
│    │    └─ReLU: 3-8                    [1, 32, 22, 27, 22]       --
│    │    └─MaxPool3d: 3-9               [1, 32, 11, 13, 11]       --
├─Dropout: 1-5                           [1, 50336]                --
├─Sequential: 1                          --                        --
│    └─Sequential: 2-4                   [1, 64]                   --
│    │    └─Linear: 3-10                 [1, 64]                   3,221,568
│    │    └─ReLU: 3-11                   [1, 64]                   --
├─Dropout: 1-6                           [1, 64]                   --
├─Sequential: 1                          --                        --
│    └─Sequential: 2-5                   [1, 128]                  --
│    │    └─Linear: 3-12                 [1, 128]                  8,320
│    │    └─ReLU: 3-13                   [1, 128]                  --
├─Linear: 1-7                            [1, 5]                    645
==========================================================================================
Total params: 3,241,389
Trainable params: 3,241,389
Non-trainable params: 0
Total mult-adds (M): 877.14
==========================================================================================
Input size (MB): 3.61
Forward/backward pass size (MB): 125.88
Params size (MB): 12.97
Estimated Total Size (MB): 142.46
==========================================================================================
"""
```

<p align="justify">
While this may seem redundant it is now compatible with the <a href="www.wandb.ai">wandb</a> package.
Specifically, this kind of configuration allows us to easily use hyperparameter sweeps with <code>wandb</code>.

## Supply your own train function to the model

To train one of these neural networks we can now use the <code>model.fit()</code> function.
You can find this function in <code>delphi.utils.train_fns</code>. In case you do not like my shipped training function,
can assign one to the network yourself.

Like so:
</p>

```python
from torch.utils.data import DataLoader, DataSet
def my_train(model, dataloader): # these two variables are mandatory if you want to supply your custom function
    # make sure that you implement a switch into training or evaluation mode if you use the same
    # function for training and evaluating
    print("This is my own train function. I can call it by using model.fit()")

my_loader = DataLoader(DataSet())
    
model = BrainStateClassifier3d(input_dims, n_classes, model_cfg, my_train)
model.fit(my_loader)
```
<p align="justify">
Even saving and loading a model is super simple:
</p>

## Save and load a model

```python
# 1. We save the state_dict of our configured model
model.save('path/to/my_configured_model')
# This creates a directory <my_configured_model> at the location <path/to>
# Within this directory you will find a <state_dict.pth> and <config.yaml> file.

# loading the model. Note that in this case you effectively create a "new" network and then fill its weights
# and biases with the state_dict you saved before. This requires you to import the model class!
loaded_model = BrainStateClassifier3d('path/to/my_configured_model')

# notice that you can still use the my_train function you supplied by calling the .fit method
loaded_model.fit(DataLoader(Dataset()))

#######################################################################################################################

# 2. We save the entirety of our configured model
model.save('path/to/my_configured_model', save_full=True)

# Using loading approach does not require you to import the SimpleLinearModel class
# in a different python file for example. 
loaded_model = torch.load('path/to/my_configured_model/model.pth')

# same here, you can still use your custom train function
print('Loaded with torch.load')
loaded_model.fit(DataLoader(Dataset()))
```

## Utility functions

<p align="justify">
I also provide some utility functions in the <code>delphi.utils</code> sub-package. You can use some of my predefined 
<code>datasets</code>, <code>layers</code>, <code>plot</code> functions, or <code>train_fns</code>. 

<code>utils.tools</code> for example contains functions to
* easily <a href="https://github.com/PhilippS893/delphi/blob/e0f3f91bef3c1e84852b5153e4ddd7f30357d344/delphi/utils/tools.py#L70">
compute_accuracy</a> scores. Simply supply your real and predicted label-vectors
* convert weights&biases config files into the format used by my code 
<a href="https://github.com/PhilippS893/delphi/blob/e0f3f91bef3c1e84852b5153e4ddd7f30357d344/delphi/utils/tools.py#L23">
convert_wandb_config</a>
* read a .yaml config file with 
<a href="https://github.com/PhilippS893/delphi/blob/e0f3f91bef3c1e84852b5153e4ddd7f30357d344/delphi/utils/tools.py#L50">
read_config</a>

In <code>utils.train_fns</code> I provide my current training and evaluation function. I can easily switch between 
training and evaluation by setting the <code>train</code> flag to <code>True/False</code>. You can find the implementation
here: <a href="https://github.com/PhilippS893/delphi/blob/e0f3f91bef3c1e84852b5153e4ddd7f30357d344/delphi/utils/train_fns.py#L11">
standard_train</a>

</p>