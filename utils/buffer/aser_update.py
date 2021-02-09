import torch
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.buffer_utils import ClassBalancedRandomSampling, random_retrieve
from utils.buffer.aser_utils import compute_knn_sv, add_minority_class_input
from utils.setup_elements import n_classes
from utils.utils import nonzero_indices, maybe_cuda


class ASER_update(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.k = params.k
        self.mem_size = params.mem_size
        self.num_tasks = params.num_tasks
        self.out_dim = n_classes[params.data]
        self.n_smp_cls = int(params.n_smp_cls)
        self.n_total_smp = int(params.n_smp_cls * self.out_dim)
        self.reservoir_update = Reservoir_update(params)
        ClassBalancedRandomSampling.class_index_cache = None

    def update(self, buffer, x, y, **kwargs):
        model = buffer.model

        place_left = self.mem_size - buffer.current_index

        # If buffer is not filled, use available space to store whole or part of batch
        if place_left:
            x_fit = x[:place_left]
            y_fit = y[:place_left]

            ind = torch.arange(start=buffer.current_index, end=buffer.current_index + x_fit.size(0), device=self.device)
            ClassBalancedRandomSampling.update_cache(buffer.buffer_label, self.out_dim,
                                                     new_y=y_fit, ind=ind, device=self.device)
            self.reservoir_update.update(buffer, x_fit, y_fit)

        # If buffer is filled, update buffer by sv
        if buffer.current_index == self.mem_size:
            # remove what is already in the buffer
            cur_x, cur_y = x[place_left:], y[place_left:]
            self._update_by_knn_sv(model, buffer, cur_x, cur_y)

    def _update_by_knn_sv(self, model, buffer, cur_x, cur_y):
        """
            Returns indices for replacement.
            Buffered instances with smallest SV are replaced by current input with higher SV.
                Args:
                    model (object): neural network.
                    buffer (object): buffer object.
                    cur_x (tensor): current input data tensor.
                    cur_y (tensor): current input label tensor.
                Returns
                    ind_buffer (tensor): indices of buffered instances to be replaced.
                    ind_cur (tensor): indices of current data to do replacement.
        """
        cur_x = maybe_cuda(cur_x)
        cur_y = maybe_cuda(cur_y)

        # Find minority class samples from current input batch
        minority_batch_x, minority_batch_y = add_minority_class_input(cur_x, cur_y, self.mem_size, self.out_dim)

        # Evaluation set
        eval_x, eval_y, eval_indices = \
            ClassBalancedRandomSampling.sample(buffer.buffer_img, buffer.buffer_label, self.n_smp_cls,
                                               device=self.device)

        # Concatenate minority class samples from current input batch to evaluation set
        eval_x = torch.cat((eval_x, minority_batch_x))
        eval_y = torch.cat((eval_y, minority_batch_y))

        # Candidate set
        cand_excl_indices = set(eval_indices.tolist())
        cand_x, cand_y, cand_ind = random_retrieve(buffer, self.n_total_smp, cand_excl_indices, return_indices=True)

        # Concatenate current input batch to candidate set
        cand_x = torch.cat((cand_x, cur_x))
        cand_y = torch.cat((cand_y, cur_y))

        sv_matrix = compute_knn_sv(model, eval_x, eval_y, cand_x, cand_y, self.k, device=self.device)
        sv = sv_matrix.sum(0)

        n_cur = cur_x.size(0)
        n_cand = cand_x.size(0)

        # Number of previously buffered instances in candidate set
        n_cand_buf = n_cand - n_cur

        sv_arg_sort = sv.argsort(descending=True)

        # Divide SV array into two segments
        # - large: candidate args to be retained; small: candidate args to be discarded
        sv_arg_large = sv_arg_sort[:n_cand_buf]
        sv_arg_small = sv_arg_sort[n_cand_buf:]

        # Extract args relevant to replacement operation
        # If current data instances are in 'large' segment, they are added to buffer
        # If buffered instances are in 'small' segment, they are discarded from buffer
        # Replacement happens between these two sets
        # Retrieve original indices from candidate args
        ind_cur = sv_arg_large[nonzero_indices(sv_arg_large >= n_cand_buf)] - n_cand_buf
        arg_buffer = sv_arg_small[nonzero_indices(sv_arg_small < n_cand_buf)]
        ind_buffer = cand_ind[arg_buffer]

        buffer.n_seen_so_far += n_cur

        # perform overwrite op
        y_upt = cur_y[ind_cur]
        x_upt = cur_x[ind_cur]
        ClassBalancedRandomSampling.update_cache(buffer.buffer_label, self.out_dim,
                                                 new_y=y_upt, ind=ind_buffer, device=self.device)
        buffer.buffer_img[ind_buffer] = x_upt
        buffer.buffer_label[ind_buffer] = y_upt
