from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from torch.utils import data
from utils.utils import maybe_cuda, AverageMeter
import torch
import copy


class Lwf(ContinualLearner):
    def __init__(self, model, opt, params):
        super(Lwf, self).__init__(model, opt, params)

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)

        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)

        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                logits = self.forward(batch_x)
                loss_old = self.kd_manager.get_kd_loss(logits, batch_x)
                loss_new = self.criterion(logits, batch_y)
                loss = 1/(self.task_seen + 1) * loss_new + (1 - 1/(self.task_seen + 1)) * loss_old
                _, pred_label = torch.max(logits, 1)
                correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                # update tracker
                acc_batch.update(correct_cnt, batch_y.size(0))
                losses_batch.update(loss, batch_y.size(0))
                # backward
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
        self.after_train()
