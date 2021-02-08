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


def add_minority_class_input(cur_x, cur_y, mem_size, num_class):
    """
    Find input instances from minority classes, and concatenate them to evaluation data/label tensors later.
    This facilitates the inclusion of minority class samples into memory when ASER's update method is used under online-class incremental setting.

    More details:

    Evaluation set may not contain any samples from minority classes (i.e., those classes with very few number of corresponding samples stored in the memory).
    This happens after task changes in online-class incremental setting.
    Minority class samples can then get very low or negative KNN-SV, making it difficult to store any of them in the memory.

    By identifying minority class samples in the current input batch, and concatenating them to the evaluation set,
        KNN-SV of the minority class samples can be artificially boosted (i.e., positive value with larger magnitude).
    This allows to quickly accomodate new class samples in the memory right after task changes.

    Threshold for being a minority class is a hyper-parameter related to the class proportion.
    In this implementation, it is randomly selected between 0 and 1 / number of all classes for each current input batch.


        Args:
            cur_x (tensor): current input data tensor.
            cur_y (tensor): current input label tensor.
            mem_size (int): memory size.
            num_class (int): number of classes in dataset.
        Returns
            minority_batch_x (tensor): subset of current input data from minority class.
            minority_batch_y (tensor): subset of current input label from minority class.
"""
    # Select input instances from minority classes that will be concatenated to pre-selected data
    threshold = torch.tensor(1).float().uniform_(0, 1 / num_class).item()

    # If number of buffered samples from certain class is lower than random threshold,
    #   that class is minority class
    cls_proportion = ClassBalancedRandomSampling.class_num_cache.float() / mem_size
    minority_ind = nonzero_indices(cls_proportion[cur_y] < threshold)

    minority_batch_x = cur_x[minority_ind]
    minority_batch_y = cur_y[minority_ind]
    return minority_batch_x, minority_batch_y
