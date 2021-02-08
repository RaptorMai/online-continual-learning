from abc import ABC, abstractmethod
import torch

from utils.utils import maybe_cuda


class Prior(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def add_expert(self):
        pass

    @abstractmethod
    def record_usage(self, usage, index=None):
        pass

    @abstractmethod
    def nl_prior(self, normalize=False):
        pass


class CumulativePrior(Prior):
    def __init__(self, params):
        super().__init__(params)
        self.log_counts = maybe_cuda(torch.tensor(
            params.log_alpha
        )).float().unsqueeze(0)

    def add_expert(self):
        self.log_counts = torch.cat(
            [self.log_counts, maybe_cuda(torch.zeros(1))],
            dim=0
        )

    def record_usage(self, usage, index=None):
        """Record expert usage

        Args:
            usage: Tensor of shape [K+1] if index is None else scalar
            index: expert index
        """
        if index is None:
            self.log_counts = torch.logsumexp(torch.stack([
                    self.log_counts,
                    usage.log()
            ], dim=1), dim=1)
        else:
            self.log_counts[index] = torch.logsumexp(torch.stack([
                self.log_counts[index],
                maybe_cuda(torch.tensor(usage)).float().log()
            ], dim=0), dim=0)

    def nl_prior(self, normalize=False):
        nl_prior = -self.log_counts
        if normalize:
            nl_prior += torch.logsumexp(self.log_counts, dim=0)
        return nl_prior

    @property
    def counts(self):
        return self.log_counts.exp()
