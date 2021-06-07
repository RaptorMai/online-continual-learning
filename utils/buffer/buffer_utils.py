import torch
import numpy as np
from utils.utils import maybe_cuda
from collections import defaultdict
from collections import Counter
import random


def random_retrieve(buffer, num_retrieve, excl_indices=None, return_indices=False):
    filled_indices = np.arange(buffer.current_index)
    if excl_indices is not None:
        excl_indices = list(excl_indices)
    else:
        excl_indices = []
    valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))
    num_retrieve = min(num_retrieve, valid_indices.shape[0])
    indices = torch.from_numpy(np.random.choice(valid_indices, num_retrieve, replace=False)).long()

    x = buffer.buffer_img[indices]

    y = buffer.buffer_label[indices]

    if return_indices:
        return x, y, indices
    else:
        return x, y


def match_retrieve(buffer, cur_y, exclud_idx=None):
    counter = Counter(cur_y.tolist())
    idx_dict = defaultdict(list)
    for idx, val in enumerate(cur_y.tolist()):
        idx_dict[val].append(idx)
    select = [None] * len(cur_y)
    for y in counter:
        idx = buffer.buffer_tracker.class_index_cache[y]
        if exclud_idx is not None:
            idx = idx - set(exclud_idx.tolist())
        if not idx or len(idx) < counter[y]:
            print('match retrieve attempt fail')
            return torch.tensor([]), torch.tensor([])
        retrieved = random.sample(list(idx), counter[y])
        for idx, val in zip(idx_dict[y], retrieved):
            select[idx] = val
    indices = torch.tensor(select)
    x = buffer.buffer_img[indices]
    y = buffer.buffer_label[indices]
    return x, y

def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    sim = torch.mm(x1, x2.t())/(w1 * w2.t()).clamp(min=eps)
    return sim


def get_grad_vector(pp, grad_dims):
    """
        gather the gradients in one vector
    """
    grads = maybe_cuda(torch.Tensor(sum(grad_dims)))
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


class ClassBalancedRandomSampling:
    # For faster label-based sampling (e.g., class balanced sampling), cache class-index via auxiliary dictionary
    # Store {class, set of memory sample indices from class} key-value pairs to speed up label-based sampling
    # e.g., {<cls_A>: {<ind_1>, <ind_2>}, <cls_B>: {}, <cls_C>: {<ind_3>}, ...}
    class_index_cache = None
    class_num_cache = None

    @classmethod
    def sample(cls, buffer_x, buffer_y, n_smp_cls, excl_indices=None, device="cpu"):
        """
            Take same number of random samples from each class from buffer.
                Args:
                    buffer_x (tensor): data buffer.
                    buffer_y (tensor): label buffer.
                    n_smp_cls (int): number of samples to take from each class.
                    excl_indices (set): indices of buffered instances to be excluded from sampling.
                    device (str): device for tensor allocation.
                Returns
                    x (tensor): class balanced random sample data tensor.
                    y (tensor): class balanced random sample label tensor.
                    sample_ind (tensor): class balanced random sample index tensor.
        """
        if excl_indices is None:
            excl_indices = set()

        # Get indices for class balanced random samples
        # cls_ind_cache = class_index_tensor_list_cache(buffer_y, num_class, excl_indices, device=device)

        sample_ind = torch.tensor([], device=device, dtype=torch.long)

        # Use cache to retrieve indices belonging to each class in buffer
        for ind_set in cls.class_index_cache.values():
            if ind_set:
                # Exclude some indices
                valid_ind = ind_set - excl_indices
                # Auxiliary indices for permutation
                perm_ind = torch.randperm(len(valid_ind), device=device)
                # Apply permutation, and select indices
                ind = torch.tensor(list(valid_ind), device=device, dtype=torch.long)[perm_ind][:n_smp_cls]
                sample_ind = torch.cat((sample_ind, ind))

        x = buffer_x[sample_ind]
        y = buffer_y[sample_ind]

        x = maybe_cuda(x)
        y = maybe_cuda(y)

        return x, y, sample_ind

    @classmethod
    def update_cache(cls, buffer_y, num_class, new_y=None, ind=None, device="cpu"):
        """
            Collect indices of buffered data from each class in set.
            Update class_index_cache with list of such sets.
                Args:
                    buffer_y (tensor): label buffer.
                    num_class (int): total number of unique class labels.
                    new_y (tensor): label tensor for replacing memory samples at ind in buffer.
                    ind (tensor): indices of memory samples to be updated.
                    device (str): device for tensor allocation.
        """
        if cls.class_index_cache is None:
            # Initialize caches
            cls.class_index_cache = defaultdict(set)
            cls.class_num_cache = torch.zeros(num_class, dtype=torch.long, device=device)

        if new_y is not None:
            # If ASER update is being used, keep updating existing caches
            # Get labels of memory samples to be replaced
            orig_y = buffer_y[ind]
            # Update caches
            for i, ny, oy in zip(ind, new_y, orig_y):
                oy_int = oy.item()
                ny_int = ny.item()
                i_int = i.item()
                # Update dictionary according to new class label of index i
                if oy_int in cls.class_index_cache and i_int in cls.class_index_cache[oy_int]:
                    cls.class_index_cache[oy_int].remove(i_int)
                    cls.class_num_cache[oy_int] -= 1
                cls.class_index_cache[ny_int].add(i_int)
                cls.class_num_cache[ny_int] += 1
        else:
            # If only ASER retrieve is being used, reset cache and update it based on buffer
            cls_ind_cache = defaultdict(set)
            for i, c in enumerate(buffer_y):
                cls_ind_cache[c.item()].add(i)
            cls.class_index_cache = cls_ind_cache


class BufferClassTracker(object):
    # For faster label-based sampling (e.g., class balanced sampling), cache class-index via auxiliary dictionary
    # Store {class, set of memory sample indices from class} key-value pairs to speed up label-based sampling
    # e.g., {<cls_A>: {<ind_1>, <ind_2>}, <cls_B>: {}, <cls_C>: {<ind_3>}, ...}

    def __init__(self, num_class, device="cpu"):
        super().__init__()
        # Initialize caches
        self.class_index_cache = defaultdict(set)
        self.class_num_cache = np.zeros(num_class)


    def update_cache(self, buffer_y, new_y=None, ind=None, ):
        """
            Collect indices of buffered data from each class in set.
            Update class_index_cache with list of such sets.
                Args:
                    buffer_y (tensor): label buffer.
                    num_class (int): total number of unique class labels.
                    new_y (tensor): label tensor for replacing memory samples at ind in buffer.
                    ind (tensor): indices of memory samples to be updated.
                    device (str): device for tensor allocation.
        """

        # Get labels of memory samples to be replaced
        orig_y = buffer_y[ind]
        # Update caches
        for i, ny, oy in zip(ind, new_y, orig_y):
            oy_int = oy.item()
            ny_int = ny.item()
            # Update dictionary according to new class label of index i
            if oy_int in self.class_index_cache and i in self.class_index_cache[oy_int]:
                self.class_index_cache[oy_int].remove(i)
                self.class_num_cache[oy_int] -= 1

            self.class_index_cache[ny_int].add(i)
            self.class_num_cache[ny_int] += 1


    def check_tracker(self):
        print(self.class_num_cache.sum())
        print(len([k for i in self.class_index_cache.values() for k in i]))