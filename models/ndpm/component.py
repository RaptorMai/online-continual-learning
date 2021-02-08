from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple

from utils.utils import maybe_cuda
from utils.global_vars import *


class Component(nn.Module, ABC):
    def __init__(self, params, experts: Tuple):
        super().__init__()
        self.params = params
        self.experts = experts

        self.optimizer = NotImplemented
        self.lr_scheduler = NotImplemented

    @abstractmethod
    def nll(self, x, y, step=None):
        """Return NLL"""
        pass

    @abstractmethod
    def collect_nll(self, x, y, step=None):
        """Return NLLs including previous experts"""
        pass

    def _clip_grad_value(self, clip_value):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_value_(group['params'], clip_value)

    def _clip_grad_norm(self, max_norm, norm_type=2):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)

    def clip_grad(self):
        clip_grad_config = MODELS_NDPM_COMPONENT_CLIP_GRAD
        if clip_grad_config['type'] == 'value':
            self._clip_grad_value(**clip_grad_config['options'])
        elif clip_grad_config['type'] == 'norm':
            self._clip_grad_norm(**clip_grad_config['options'])
        else:
            raise ValueError('Invalid clip_grad type: {}'
                             .format(clip_grad_config['type']))

    @staticmethod
    def build_optimizer(optim_config, params):
        return getattr(torch.optim, optim_config['type'])(
            params, **optim_config['options'])

    @staticmethod
    def build_lr_scheduler(lr_config, optimizer):
        return getattr(torch.optim.lr_scheduler, lr_config['type'])(
            optimizer, **lr_config['options'])

    def weight_decay_loss(self):
        loss = maybe_cuda(torch.zeros([]))
        for param in self.parameters():
            loss += torch.norm(param) ** 2
        return loss


class ComponentG(Component, ABC):
    def setup_optimizer(self):
        self.optimizer = self.build_optimizer(
            {'type': self.params.optimizer, 'options': {'lr': self.params.learning_rate}}, self.parameters())
        self.lr_scheduler = self.build_lr_scheduler(
            MODELS_NDPM_COMPONENT_LR_SCHEDULER_G, self.optimizer)

    def collect_nll(self, x, y=None, step=None):
        """Default `collect_nll`

        Warning: Parameter-sharing components should implement their own
            `collect_nll`

        Returns:
            nll: Tensor of shape [B, 1+K]
        """
        outputs = [expert.g.nll(x, y, step) for expert in self.experts]
        nll = outputs
        output = self.nll(x, y, step)
        nll.append(output)
        return torch.stack(nll, dim=1)



class ComponentD(Component, ABC):
    def setup_optimizer(self):
        self.optimizer = self.build_optimizer(
            {'type': self.params.optimizer, 'options': {'lr': self.params.learning_rate}}, self.parameters())
        self.lr_scheduler = self.build_lr_scheduler(
            MODELS_NDPM_COMPONENT_LR_SCHEDULER_D, self.optimizer)

    def collect_forward(self, x):
        """Default `collect_forward`

        Warning: Parameter-sharing components should implement their own
            `collect_forward`

        Returns:
            output: Tensor of shape [B, 1+K, C]
        """
        outputs = [expert.d(x) for expert in self.experts]
        outputs.append(self.forward(x))
        return torch.stack(outputs, 1)

    def collect_nll(self, x, y, step=None):
        """Default `collect_nll`

        Warning: Parameter-sharing components should implement their own
            `collect_nll`

        Returns:
            nll: Tensor of shape [B, 1+K]
        """
        outputs = [expert.d.nll(x, y, step) for expert in self.experts]
        nll = outputs
        output = self.nll(x, y, step)
        nll.append(output)
        return torch.stack(nll, dim=1)
