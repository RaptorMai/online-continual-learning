import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy


def gaussian_nll(x, mean, log_var, min_noise=0.001):
    return (
        ((x - mean) ** 2 + min_noise) / (2 * log_var.exp() + 1e-8)
        + 0.5 * log_var + 0.5 * np.log(2 * np.pi)
    )


def laplace_nll(x, median, log_scale, min_noise=0.01):
    return (
        ((x - median).abs() + min_noise) / (log_scale.exp() + 1e-8)
        + log_scale + np.log(2)
    )


def bernoulli_nll(x, p):
    # Broadcast
    x_exp, p_exp = [], []
    for x_size, p_size in zip(x.size(), p.size()):
        if x_size > p_size:
            x_exp.append(-1)
            p_exp.append(x_size)
        elif x_size < p_size:
            x_exp.append(p_size)
            p_exp.append(-1)
        else:
            x_exp.append(-1)
            p_exp.append(-1)
    x = x.expand(*x_exp)
    p = p.expand(*p_exp)

    return binary_cross_entropy(p, x, reduction='none')


def logistic_nll(x, mean, log_scale):
    bin_size = 1 / 256
    scale = log_scale.exp()
    x_centered = x - mean
    cdf1 = x_centered / scale
    cdf2 = (x_centered + bin_size) / scale
    p = torch.sigmoid(cdf2) - torch.sigmoid(cdf1) + 1e-12
    return -p.log()
