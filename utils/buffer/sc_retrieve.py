from utils.buffer.buffer_utils import match_retrieve
import torch

class Match_retrieve(object):
    def __init__(self, params):
        super().__init__()
        self.num_retrieve = params.eps_mem_batch
        self.warmup = params.warmup

    def retrieve(self, buffer, **kwargs):
        if buffer.n_seen_so_far > self.num_retrieve * self.warmup:
            cur_x, cur_y = kwargs['x'], kwargs['y']
            return match_retrieve(buffer, cur_y)
        else:
            return torch.tensor([]), torch.tensor([])