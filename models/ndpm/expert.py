import torch.nn as nn

from models.ndpm.classifier import ResNetSharingClassifier
from models.ndpm.vae import CnnSharingVae
from utils.utils import maybe_cuda

from utils.global_vars import *


class Expert(nn.Module):
    def __init__(self, params, experts=()):
        super().__init__()
        self.id = len(experts)
        self.experts = experts

        self.g = maybe_cuda(CnnSharingVae(params, experts))
        self.d = maybe_cuda(ResNetSharingClassifier(params, experts)) if not MODELS_NDPM_NDPM_DISABLE_D else None


        # use random initialized g if it's a placeholder
        if self.id == 0:
            self.eval()
            for p in self.g.parameters():
                p.requires_grad = False

        # use random initialized d if it's a placeholder
        if self.id == 0 and self.d is not None:
            for p in self.d.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.d(x)

    def nll(self, x, y, step=None):
        """Negative log likelihood"""
        nll = self.g.nll(x, step)
        if self.d is not None:
            d_nll = self.d.nll(x, y, step)
            nll = nll + d_nll
        return nll

    def collect_nll(self, x, y, step=None):
        if self.id == 0:
            nll = self.nll(x, y, step)
            return nll.unsqueeze(1)

        nll = self.g.collect_nll(x, step)
        if self.d is not None:
            d_nll = self.d.collect_nll(x, y, step)
            nll = nll + d_nll

        return nll

    def lr_scheduler_step(self):
        if self.g.lr_scheduler is not NotImplemented:
            self.g.lr_scheduler.step()
        if self.d is not None and self.d.lr_scheduler is not NotImplemented:
            self.d.lr_scheduler.step()

    def clip_grad(self):
        self.g.clip_grad()
        if self.d is not None:
            self.d.clip_grad()

    def optimizer_step(self):
        self.g.optimizer.step()
        if self.d is not None:
            self.d.optimizer.step()
