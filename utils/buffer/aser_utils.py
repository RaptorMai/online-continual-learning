import torch
from utils.utils import maybe_cuda, mini_batch_deep_features, euclidean_distance, nonzero_indices, ohe_label
from utils.setup_elements import n_classes
from utils.buffer.buffer_utils import ClassBalancedRandomSampling


def compute_knn_sv(model, eval_x, eval_y, cand_x, cand_y, k, device="cpu"):
    """
        Compute KNN SV of candidate data w.r.t. evaluation data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                eval_y (tensor): evaluation label tensor.
                cand_x (tensor): candidate data tensor.
                cand_y (tensor): candidate label tensor.
                k (int): number of nearest neighbours.
                device (str): device for tensor allocation.
            Returns
                sv_matrix (tensor): KNN Shapley value matrix of candidate data w.r.t. evaluation data.
    """
    # Compute KNN SV score for candidate samples w.r.t. evaluation samples
    n_eval = eval_x.size(0)
    n_cand = cand_x.size(0)
    # Initialize SV matrix to matrix of -1
    sv_matrix = torch.zeros((n_eval, n_cand), device=device)
    # Get deep features
    eval_df, cand_df = deep_features(model, eval_x, n_eval, cand_x, n_cand)
    # Sort indices based on distance in deep feature space
    sorted_ind_mat = sorted_cand_ind(eval_df, cand_df, n_eval, n_cand)

    # Evaluation set labels
    el = eval_y
    el_vec = el.repeat([n_cand, 1]).T
    # Sorted candidate set labels
    cl = cand_y[sorted_ind_mat]

    # Indicator function matrix
    indicator = (el_vec == cl).float()
    indicator_next = torch.zeros_like(indicator, device=device)
    indicator_next[:, 0:n_cand - 1] = indicator[:, 1:]
    indicator_diff = indicator - indicator_next

    cand_ind = torch.arange(n_cand, dtype=torch.float, device=device) + 1
    denom_factor = cand_ind.clone()
    denom_factor[:n_cand - 1] = denom_factor[:n_cand - 1] * k
    numer_factor = cand_ind.clone()
    numer_factor[k:n_cand - 1] = k
    numer_factor[n_cand - 1] = 1
    factor = numer_factor / denom_factor

    indicator_factor = indicator_diff * factor
    indicator_factor_cumsum = indicator_factor.flip(1).cumsum(1).flip(1)

    # Row indices
    row_ind = torch.arange(n_eval, device=device)
    row_mat = torch.repeat_interleave(row_ind, n_cand).reshape([n_eval, n_cand])

    # Compute SV recursively
    sv_matrix[row_mat, sorted_ind_mat] = indicator_factor_cumsum

    return sv_matrix


def deep_features(model, eval_x, n_eval, cand_x, n_cand):
    """
        Compute deep features of evaluation and candidate data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                n_eval (int): number of evaluation data.
                cand_x (tensor): candidate data tensor.
                n_cand (int): number of candidate data.
            Returns
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
    """
    # Get deep features
    if cand_x is None:
        num = n_eval
        total_x = eval_x
    else:
        num = n_eval + n_cand
        total_x = torch.cat((eval_x, cand_x), 0)

    # compute deep features with mini-batches
    total_x = maybe_cuda(total_x)
    deep_features_ = mini_batch_deep_features(model, total_x, num)

    eval_df = deep_features_[0:n_eval]
    cand_df = deep_features_[n_eval:]
    return eval_df, cand_df


def sorted_cand_ind(eval_df, cand_df, n_eval, n_cand):
    """
        Sort indices of candidate data according to
            their Euclidean distance to each evaluation data in deep feature space.
            Args:
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
                n_eval (int): number of evaluation data.
                n_cand (int): number of candidate data.
            Returns
                sorted_cand_ind (tensor): sorted indices of candidate set w.r.t. each evaluation data.
    """
    # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
    # Preprocess feature vectors to facilitate vector-wise distance computation
    eval_df_repeat = eval_df.repeat([1, n_cand]).reshape([n_eval * n_cand, eval_df.shape[1]])
    cand_df_tile = cand_df.repeat([n_eval, 1])
    # Compute distance between evaluation and candidate feature vectors
    distance_vector = euclidean_distance(eval_df_repeat, cand_df_tile)
    # Turn distance vector into distance matrix
    distance_matrix = distance_vector.reshape((n_eval, n_cand))
    # Sort candidate set indices based on distance
    sorted_cand_ind_ = distance_matrix.argsort(1)
    return sorted_cand_ind_


class AuxSamplingManager:
    def __init__(self, params):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mem_size = params.mem_size
        self.num_tasks = params.num_tasks
        self.out_dim = n_classes[params.data]
        self.aux_smp_type = params.aux_smp_type

        # If auxiliary sampling manager type is "minority"
        if self.aux_smp_type == "minority":
            # Expected minimum number of samples per class
            self.min_exp_n_per_cls = self.mem_size // self.out_dim

        # If auxiliary sampling manager type is "accum"
        if self.aux_smp_type == "accum":
            # Number of unique class labels in split task
            self.n_labels_per_split_task = self.out_dim // self.num_tasks
            # All unique class labels in current task
            self.cur_task_label_set = set()
            # Number of current task labels seen so far
            self.count_cur = 0
            # Number of current task samples to be freely entered into memory from first encounter
            self.accum_size = self.mem_size // self.out_dim * self.n_labels_per_split_task

    def get_aux_samples(self, cur_x, cur_y, threshold):
        """
            Apply auxiliary sampling methods for ASER update to promote class balance in buffer.
            Return evaluation set and candidate set for ASER update.

            In New Class scenario, this boosts SV of first few new task data.
            Once their number exceeds threshold, manager stops.
            Buffer is then class balanced, and fair SV comparison can be made afterwards.

            See accum_update and add_minority_class_input methods for details of accum and minority options.
                Args:
                    cur_x (tensor): current input data tensor.
                    cur_y (tensor): current input label tensor.
                Returns
                    minority_batch_x (tensor): subset of current input data from minority class.
                                                If minority is not selected, return None.
                    minority_batch_y (tensor): subset of current input label from minority class.
                                                If minority is not selected, return None.
                    accum_excl_indices (set): indices of buffered instances from current task.
                                                If accum is selected and activated, they are excluded from sampling.
                                                Otherwise, return empty set.
        """
        accum_excl_indices = set()
        minority_batch_x, minority_batch_y = None, None
        # Run auxiliary sampling methods
        if self.aux_smp_type == "accum":
            accum_excl_indices = self.accum_update(cur_y)
        else:
            minority_batch_x, minority_batch_y = self.add_minority_class_input(cur_x, cur_y, threshold)

        return minority_batch_x, minority_batch_y, accum_excl_indices

    def accum_update(self, cur_y):
        """
            Apply accum by preventing current task data instances from being added to evaluation and candidate set
                such that evaluation and candidate set are solely from previous task.
            End Goal of Accum:
                After soring SV, replace candidate data with smallest SV with current input.
                This accumulates current task data in buffer
                    as long as current task data count is smaller than threshold specified by accum_size.
                This promotes inclusion of new split task samples so class balance in buffer can be achieved later.
                Args:
                    cur_y (tensor): current input label tensor.
                Returns
                    accum_excl_indices (set): updated indices of buffered instances to be excluded from sampling.
        """
        excl_indices = set()
        if self.aux_smp_type == "accum":
            # Distinct current input labels
            cur_label_set = set(cur_y.tolist())

            # If current input is new split task, update current task label set and count
            self.check_new_split_task(cur_label_set)

            # accum activation condition
            # Number of current task samples seen so far (i.e., self.count_cur)
            #   should be less than threshold (i.e., self.accum_size)
            if self.count_cur <= self.accum_size:
                # Update count
                self.count_cur += cur_y.size(0)
                cls_ind_cache = ClassBalancedRandomSampling.class_index_cache

                for c in cur_label_set:
                    excl_indices = excl_indices.union(cls_ind_cache[c])

        return excl_indices

    def check_new_split_task(self, cur_label_set):
        """
            Check if new split task has started. If so, update current task label set and count.
                Args:
                    cur_label_set (set): set of distinct current input labels.
        """
        # Check if task has changed
        # If current input labels are not subset of current task label set
        # Current task label set should be updated (i.e., it may be no longer 'current' task label set)
        if not cur_label_set.issubset(self.cur_task_label_set):
            if len(self.cur_task_label_set) < self.n_labels_per_split_task:
                # Current task label set is partially updated, so update it further with current input labels
                self.cur_task_label_set = self.cur_task_label_set.union(cur_label_set)
            else:
                # Task has changed. Replace current task label set with current input labels set
                # Reset current task count
                self.cur_task_label_set = cur_label_set
                self.count_cur = 0

    def add_minority_class_input(self, cur_x, cur_y, threshold):
        """
            Find input instances from minority classes, and concatenate them to pre-selected data/label tensors.
                Args:
                    cur_x (tensor): current input data tensor.
                    cur_y (tensor): current input label tensor.
                Returns
                    minority_batch_x (tensor): subset of current input data from minority class.
                    minority_batch_y (tensor): subset of current input label from minority class.
        """
        # Select input instances from minority classes that will be concatenated to pre-selected data
        if threshold == 'rand':
            threshold = torch.tensor(1).float().uniform_(0, self.min_exp_n_per_cls).int()
        else:
            threshold = self.min_exp_n_per_cls
        # If number of buffered samples from certain class is lower than random threshold,
        #   that class is minority class
        cls_num_cache = ClassBalancedRandomSampling.class_num_cache
        minority_class = nonzero_indices(cls_num_cache < threshold)
        # Get minority class mask
        minority_class_mask = torch.zeros(self.out_dim, device=self.device).byte()
        minority_class_mask[minority_class] = 1
        # Get one-hot-encoding of current input label
        cur_y_ohe = ohe_label(cur_y, self.out_dim, device=self.device)
        # Apply minority class mask to one-hot-encoded current input label
        #   to get indices of current input from minority classes
        ind_minority_cur_y = nonzero_indices((cur_y_ohe & minority_class_mask).sum(1))
        minority_batch_x = cur_x[ind_minority_cur_y]
        minority_batch_y = cur_y[ind_minority_cur_y]
        return minority_batch_x, minority_batch_y
