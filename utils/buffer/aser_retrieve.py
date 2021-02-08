import torch
from utils.buffer.buffer_utils import random_retrieve, ClassBalancedRandomSampling
from utils.buffer.aser_utils import compute_knn_sv
from utils.utils import maybe_cuda
from utils.setup_elements import n_classes


class ASER_retrieve(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.num_retrieve = params.eps_mem_batch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.k = params.k
        self.mem_size = params.mem_size
        self.aser_type = params.aser_type
        self.n_smp_cls = int(params.n_smp_cls)
        self.out_dim = n_classes[params.data]
        self.is_aser_upt = params.update == "ASER"
        ClassBalancedRandomSampling.class_index_cache = None

    def retrieve(self, buffer, **kwargs):
        model = buffer.model

        if buffer.n_seen_so_far <= self.mem_size:
            # Use random retrieval until buffer is filled
            ret_x, ret_y = random_retrieve(buffer, self.num_retrieve)
        else:
            # Use ASER retrieval if buffer is filled
            cur_x, cur_y = kwargs['x'], kwargs['y']
            buffer_x, buffer_y = buffer.buffer_img, buffer.buffer_label
            ret_x, ret_y = self._retrieve_by_knn_sv(model, buffer_x, buffer_y, cur_x, cur_y, self.num_retrieve)
        return ret_x, ret_y

    def _retrieve_by_knn_sv(self, model, buffer_x, buffer_y, cur_x, cur_y, num_retrieve):
        """
            Retrieves data instances with top-N Shapley Values from candidate set.
                Args:
                    model (object): neural network.
                    buffer_x (tensor): data buffer.
                    buffer_y (tensor): label buffer.
                    cur_x (tensor): current input data tensor.
                    cur_y (tensor): current input label tensor.
                    num_retrieve (int): number of data instances to be retrieved.
                Returns
                    ret_x (tensor): retrieved data tensor.
                    ret_y (tensor): retrieved label tensor.
        """
        cur_x = maybe_cuda(cur_x)
        cur_y = maybe_cuda(cur_y)

        # Reset and update ClassBalancedRandomSampling cache if ASER update is not enabled
        if not self.is_aser_upt:
            ClassBalancedRandomSampling.update_cache(buffer_y, self.out_dim)

        # Get candidate data for retrieval (i.e., cand <- class balanced subsamples from memory)
        cand_x, cand_y, cand_ind = \
            ClassBalancedRandomSampling.sample(buffer_x, buffer_y, self.n_smp_cls, device=self.device)

        # Type 1 - Adversarial SV
        # Get evaluation data for type 1 (i.e., eval <- current input)
        eval_adv_x, eval_adv_y = cur_x, cur_y
        # Compute adversarial Shapley value of candidate data
        # (i.e., sv wrt current input)
        sv_matrix_adv = compute_knn_sv(model, eval_adv_x, eval_adv_y, cand_x, cand_y, self.k, device=self.device)

        if self.aser_type != "neg_sv":
            # Type 2 - Cooperative SV
            # Get evaluation data for type 2
            # (i.e., eval <- class balanced subsamples from memory excluding those already in candidate set)
            excl_indices = set(cand_ind.tolist())
            eval_coop_x, eval_coop_y, _ = \
                ClassBalancedRandomSampling.sample(buffer_x, buffer_y, self.n_smp_cls,
                                                   excl_indices=excl_indices, device=self.device)
            # Compute Shapley value
            sv_matrix_coop = \
                compute_knn_sv(model, eval_coop_x, eval_coop_y, cand_x, cand_y, self.k, device=self.device)
            if self.aser_type == "asv":
                # Use extremal SVs for computation
                sv = sv_matrix_coop.max(0).values - sv_matrix_adv.min(0).values
            else:
                # Use mean variation for aser_type == "asvm" or anything else
                sv = sv_matrix_coop.mean(0) - sv_matrix_adv.mean(0)
        else:
            # aser_type == "neg_sv"
            # No Type 1 - Cooperative SV; Use sum of Adversarial SV only
            sv = sv_matrix_adv.sum(0) * -1

        ret_ind = sv.argsort(descending=True)

        ret_x = cand_x[ret_ind][:num_retrieve]
        ret_y = cand_y[ret_ind][:num_retrieve]
        return ret_x, ret_y
