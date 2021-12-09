import glob
import os
from operator import itemgetter
from itertools import chain
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from rl.envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def masked_mean(x, mask, dim=-1, keepdim=False):
    assert x.shape == mask.shape
    return torch.sum(x * mask.float(), dim=dim, keepdim=keepdim) / torch.sum(mask, dim=dim, keepdim=keepdim)


def masked_sum(x, mask, dim=-1, keepdim=False):
    assert x.shape == mask.shape
    return torch.sum(x * mask.float(), dim=dim, keepdim=keepdim)


def create_hook(name):
    def hook(grad):
        print(name, "nan-grad: ", np.isnan(grad.cpu().numpy()).sum())
    return hook


def count_parameters(net):
    """ Returns total number of trainable parameters in net """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _list_to_sequence(x, indices):
    return torch.nn.utils.rnn.pack_sequence(list(chain.from_iterable(itemgetter(*indices)(x))), enforce_sorted=False)