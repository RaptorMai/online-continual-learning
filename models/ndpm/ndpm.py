import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from utils.utils import maybe_cuda
from utils.global_vars import *
from .expert import Expert
from .priors import CumulativePrior


class Ndpm(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.experts = nn.ModuleList([Expert(params)])
        self.stm_capacity = params.stm_capacity
        self.stm_x = []
        self.stm_y = []
        self.prior = CumulativePrior(params)

    def get_experts(self):
        return tuple(self.experts.children())

    def forward(self, x):
        with torch.no_grad():
            if len(self.experts) == 1:
                raise RuntimeError('There\'s no expert to run on the input')
            x = maybe_cuda(x)
            log_evid = -self.experts[-1].g.collect_nll(x)  # [B, 1+K]
            log_evid = log_evid[:, 1:].unsqueeze(2)  # [B, K, 1]
            log_prior = -self.prior.nl_prior()[1:]  # [K]
            log_prior -= torch.logsumexp(log_prior, dim=0)
            log_prior = log_prior.unsqueeze(0).unsqueeze(2)  # [1, K, 1]
            log_joint = log_prior + log_evid  # [B, K, 1]
            if not MODELS_NDPM_NDPM_DISABLE_D:
                log_pred = self.experts[-1].d.collect_forward(x)  # [B, 1+K, C]
                log_pred = log_pred[:, 1:, :]  # [B, K, C]
                log_joint = log_joint + log_pred  # [B, K, C]

            log_joint = log_joint.logsumexp(dim=1).squeeze()  # [B,] or [B, C]
            return log_joint


    def learn(self, x, y):
        x, y = maybe_cuda(x), maybe_cuda(y)

        if MODELS_NDPM_NDPM_SEND_TO_STM_ALWAYS:
            self.stm_x.extend(torch.unbind(x.cpu()))
            self.stm_y.extend(torch.unbind(y.cpu()))
        else:
            # Determine the destination of each data point
            nll = self.experts[-1].collect_nll(x, y)  # [B, 1+K]
            nl_prior = self.prior.nl_prior()  # [1+K]
            nl_joint = nll + nl_prior.unsqueeze(0).expand(
                nll.size(0), -1)  # [B, 1+K]

            # Save to short-term memory
            destination = maybe_cuda(torch.argmin(nl_joint, dim=1))  # [B]
            to_stm = destination == 0  # [B]
            self.stm_x.extend(torch.unbind(x[to_stm].cpu()))
            self.stm_y.extend(torch.unbind(y[to_stm].cpu()))

            # Train expert
            with torch.no_grad():
                min_joint = nl_joint.min(dim=1)[0].view(-1, 1)
                to_expert = torch.exp(-nl_joint + min_joint)  # [B, 1+K]
                to_expert[:, 0] = 0.  # [B, 1+K]
                to_expert = \
                    to_expert / (to_expert.sum(dim=1).view(-1, 1) + 1e-7)

            # Compute losses per expert
            nll_for_train = nll * (1. - to_stm.float()).unsqueeze(1)  # [B,1+K]
            losses = (nll_for_train * to_expert).sum(0)  # [1+K]

            # Record expert usage
            expert_usage = to_expert.sum(dim=0)  # [K+1]
            self.prior.record_usage(expert_usage)

            # Do lr_decay implicitly
            if MODELS_NDPM_NDPM_IMPLICIT_LR_DECAY:
                losses = losses \
                         * self.params.stm_capacity / (self.prior.counts + 1e-8)
            loss = losses.sum()

            if loss.requires_grad:
                update_threshold = 0
                for k, usage in enumerate(expert_usage):
                    if usage > update_threshold:
                        self.experts[k].zero_grad()
                loss.backward()
                for k, usage in enumerate(expert_usage):
                    if usage > update_threshold:
                        self.experts[k].clip_grad()
                        self.experts[k].optimizer_step()
                        self.experts[k].lr_scheduler_step()

        # Sleep
        if len(self.stm_x) >= self.stm_capacity:
            dream_dataset = TensorDataset(
                torch.stack(self.stm_x), torch.stack(self.stm_y))
            self.sleep(dream_dataset)
            self.stm_x = []
            self.stm_y = []

    def sleep(self, dream_dataset):
        print('\nGoing to sleep...')
        # Add new expert and optimizer
        expert = Expert(self.params, self.get_experts())
        self.experts.append(expert)
        self.prior.add_expert()

        stacked_stm_x = torch.stack(self.stm_x)
        stacked_stm_y = torch.stack(self.stm_y)
        indices = torch.randperm(stacked_stm_x.size(0))
        train_size = stacked_stm_x.size(0) - MODELS_NDPM_NDPM_SLEEP_SLEEP_VAL_SIZE
        dream_dataset = TensorDataset(
            stacked_stm_x[indices[:train_size]],
            stacked_stm_y[indices[:train_size]])

        # Prepare data iterator
        self.prior.record_usage(len(dream_dataset), index=-1)
        dream_iterator = iter(DataLoader(
            dream_dataset,
            batch_size=MODELS_NDPM_NDPM_SLEEP_BATCH_SIZE,
            num_workers=MODELS_NDPM_NDPM_SLEEP_NUM_WORKERS,
            sampler=RandomSampler(
                dream_dataset,
                replacement=True,
                num_samples=(
                        MODELS_NDPM_NDPM_SLEEP_STEP_G *
                        MODELS_NDPM_NDPM_SLEEP_BATCH_SIZE
                ))
        ))

        # Train generative component
        for step, (x, y) in enumerate(dream_iterator):
            step += 1
            x, y = maybe_cuda(x), maybe_cuda(y)
            g_loss = expert.g.nll(x, y, step=step)
            g_loss = (g_loss + MODELS_NDPM_NDPM_WEIGHT_DECAY
                      * expert.g.weight_decay_loss())
            expert.g.zero_grad()
            g_loss.mean().backward()
            expert.g.clip_grad()
            expert.g.optimizer.step()

            if step % MODELS_NDPM_NDPM_SLEEP_SUMMARY_STEP == 0:
                print('\r   [Sleep-G %6d] loss: %5.1f' % (
                    step, g_loss.mean()
                ), end='')
        print()

        dream_iterator = iter(DataLoader(
            dream_dataset,
            batch_size=MODELS_NDPM_NDPM_SLEEP_BATCH_SIZE,
            num_workers=MODELS_NDPM_NDPM_SLEEP_NUM_WORKERS,
            sampler=RandomSampler(
                dream_dataset,
                replacement=True,
                num_samples=(
                        MODELS_NDPM_NDPM_SLEEP_STEP_D *
                        MODELS_NDPM_NDPM_SLEEP_BATCH_SIZE)
            )
        ))

        # Train discriminative component
        if not MODELS_NDPM_NDPM_DISABLE_D:
            for step, (x, y) in enumerate(dream_iterator):
                step += 1
                x, y = maybe_cuda(x), maybe_cuda(y)
                d_loss = expert.d.nll(x, y, step=step)
                d_loss = (d_loss + MODELS_NDPM_NDPM_WEIGHT_DECAY
                          * expert.d.weight_decay_loss())
                expert.d.zero_grad()
                d_loss.mean().backward()
                expert.d.clip_grad()
                expert.d.optimizer.step()

                if step % MODELS_NDPM_NDPM_SLEEP_SUMMARY_STEP == 0:
                    print('\r   [Sleep-D %6d] loss: %5.1f' % (
                        step, d_loss.mean()
                    ), end='')

        expert.lr_scheduler_step()
        expert.lr_scheduler_step()
        expert.eval()
        print()

    @staticmethod
    def _nl_joint(nl_prior, nll):
        batch = nll.size(0)
        nl_prior = nl_prior.unsqueeze(0).expand(batch, -1)  # [B, 1+K]
        return nll + nl_prior

    def train(self, mode=True):
        # Disabled
        pass
