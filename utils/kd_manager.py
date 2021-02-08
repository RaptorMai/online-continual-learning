import copy
import torch
from torch.nn import functional as F


def loss_fn_kd(scores, target_scores, T=2.):
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)
    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    kd_loss = (-1 * targets_norm * log_scores_norm).sum(dim=1).mean() * T ** 2
    return kd_loss


class KdManager:
    def __init__(self):
        self.teacher_model = None

    def update_teacher(self, model):
        self.teacher_model = copy.deepcopy(model)

    def get_kd_loss(self, cur_model_logits, x):
        if self.teacher_model is not None:
            with torch.no_grad():
                prev_model_logits = self.teacher_model.forward(x)
            dist_loss = loss_fn_kd(cur_model_logits, prev_model_logits)
        else:
            dist_loss = 0
        return dist_loss
