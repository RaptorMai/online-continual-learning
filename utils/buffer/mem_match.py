from utils.buffer.buffer_utils import match_retrieve
from utils.buffer.buffer_utils import random_retrieve
import torch

class MemMatch_retrieve(object):
    def __init__(self, params):
        super().__init__()
        self.num_retrieve = params.eps_mem_batch
        self.warmup = params.warmup


    def retrieve(self, buffer, **kwargs):
        match_x, match_y = torch.tensor([]), torch.tensor([])
        candidate_x, candidate_y = torch.tensor([]), torch.tensor([])
        if buffer.n_seen_so_far > self.num_retrieve * self.warmup:
            while match_x.size(0) == 0:
                candidate_x, candidate_y, indices = random_retrieve(buffer, self.num_retrieve,return_indices=True)
                if candidate_x.size(0) == 0:
                    return  candidate_x, candidate_y, match_x, match_y
                match_x, match_y = match_retrieve(buffer, candidate_y, indices)
        return candidate_x, candidate_y, match_x, match_y