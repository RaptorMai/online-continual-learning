import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, f=None):
        super().__init__()
        self.f = f if f is not None else (lambda x: x)

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)
