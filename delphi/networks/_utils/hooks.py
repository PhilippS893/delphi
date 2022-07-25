import numpy as np

_activation = {}


def get_activation(name):
    def hook(model, input, output):
        _activation[name] = output.detach()

    return hook


def hook_net(net):
    for seq_name, m in net.named_children():
        if seq_name in ['conv', 'lin']:
            for layer_name, l in m.named_children():
                l[0].register_forward_hook(get_activation(layer_name))
        elif seq_name == 'out':
            # print(seq_name, m)
            m.register_forward_hook(get_activation(seq_name))
        else:
            continue


def assign_activations(this_dict, activations=None):
    if activations is None:
        activations = _activation
    for i, key in enumerate(activations.keys()):
        if key not in this_dict:
            this_dict[key] = activations[key].cpu().numpy()
        else:
            this_dict[key] = np.vstack((this_dict[key], activations[key].cpu().numpy()))

    return this_dict
