from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from models.ndpm.ndpm import Ndpm
from utils.setup_elements import transforms_match
from torch.utils import data
from utils.utils import maybe_cuda, AverageMeter
import torch


class Cndpm(ContinualLearner):
    def __init__(self, model, opt, params):
        super(Cndpm, self).__init__(model, opt, params)
        self.model = model


    def train_learner(self, x_train, y_train):
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # setup tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()

        self.model.train()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                self.model.learn(batch_x, batch_y)
                if self.params.verbose:
                    print('\r[Step {:4}] STM: {:5}/{} | #Expert: {}'.format(
                        i,
                        len(self.model.stm_x), self.params.stm_capacity,
                        len(self.model.experts) - 1
                    ), end='')
        print()
