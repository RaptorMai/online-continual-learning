from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import maybe_cuda
from .component import ComponentD
from utils.global_vars import *
from utils.setup_elements import n_classes


class Classifier(ComponentD, ABC):
    def __init__(self, params, experts):
        super().__init__(params, experts)
        self.ce_loss = nn.NLLLoss(reduction='none')

    @abstractmethod
    def forward(self, x):
        """Output log P(y|x)"""
        pass

    def nll(self, x, y, step=None):
        x, y = maybe_cuda(x), maybe_cuda(y)
        log_softmax = self.forward(x)
        loss_pred = self.ce_loss(log_softmax, y)

        # Classifier chilling
        chilled_log_softmax = F.log_softmax(
            log_softmax / self.params.classifier_chill, dim=1)
        chilled_loss_pred = self.ce_loss(chilled_log_softmax, y)

        # Value with chill & gradient without chill
        loss_pred = loss_pred - loss_pred.detach() \
            + chilled_loss_pred.detach()

        return loss_pred


class SharingClassifier(Classifier, ABC):
    @abstractmethod
    def forward(self, x, collect=False):
        pass

    def collect_forward(self, x):
        dummy_pred = self.experts[0](x)
        preds, _ = self.forward(x, collect=True)
        return torch.stack([dummy_pred] + preds, dim=1)

    def collect_nll(self, x, y, step=None):
        preds = self.collect_forward(x)  # [B, 1+K, C]
        loss_preds = []
        for log_softmax in preds.unbind(dim=1):
            loss_pred = self.ce_loss(log_softmax, y)

            # Classifier chilling
            chilled_log_softmax = F.log_softmax(
                log_softmax / self.params.classifier_chill, dim=1)
            chilled_loss_pred = self.ce_loss(chilled_log_softmax, y)

            # Value with chill & gradient without chill
            loss_pred = loss_pred - loss_pred.detach() \
                        + chilled_loss_pred.detach()

            loss_preds.append(loss_pred)
        return torch.stack(loss_preds, dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, upsample=None,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        transpose = upsample is not None and stride != 1
        self.conv1 = (
            conv4x4t(inplanes, planes, stride) if transpose else
            conv3x3(inplanes, planes, stride)
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        elif self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv4x4t(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """4x4 transposed convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes, out_planes, kernel_size=4, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class ResNetSharingClassifier(SharingClassifier):
    block = BasicBlock
    num_blocks = [2, 2, 2, 2]
    norm_layer = nn.InstanceNorm2d

    def __init__(self, params, experts):
        super().__init__(params, experts)
        self.precursors = [expert.d for expert in self.experts[1:]]
        first = len(self.precursors) == 0

        if MODELS_NDPM_CLASSIFIER_NUM_BLOCKS is not None:
            num_blocks = MODELS_NDPM_CLASSIFIER_NUM_BLOCKS
        else:
            num_blocks = self.num_blocks
        if MODELS_NDPM_CLASSIFIER_NORM_LAYER is not None:
            self.norm_layer = getattr(nn, MODELS_NDPM_CLASSIFIER_NORM_LAYER)
        else:
            self.norm_layer = nn.BatchNorm2d

        num_classes = n_classes[params.data]
        nf = MODELS_NDPM_CLASSIFIER_CLS_NF_BASE if first else MODELS_NDPM_CLASSIFIER_CLS_NF_EXT
        nf_cat = MODELS_NDPM_CLASSIFIER_CLS_NF_BASE \
            + len(self.precursors) * MODELS_NDPM_CLASSIFIER_CLS_NF_EXT
        self.nf = MODELS_NDPM_CLASSIFIER_CLS_NF_BASE if first else MODELS_NDPM_CLASSIFIER_CLS_NF_EXT
        self.nf_cat = nf_cat

        self.layer0 = nn.Sequential(
            nn.Conv2d(
                3, nf * 1, kernel_size=3, stride=1, padding=1, bias=False
            ),
            self.norm_layer(nf * 1),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(
            nf_cat * 1, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            nf_cat * 1, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            nf_cat * 2, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(
            nf_cat * 4, nf * 8, num_blocks[3], stride=2)
        self.predict = nn.Sequential(
            nn.Linear(nf_cat * 8, num_classes),
            nn.LogSoftmax(dim=1)
        )
        self.setup_optimizer()

    def _make_layer(self, nf_in, nf_out, num_blocks, stride):
        norm_layer = self.norm_layer
        block = self.block
        downsample = None
        if stride != 1 or nf_in != nf_out:
            downsample = nn.Sequential(
                conv1x1(nf_in, nf_out, stride),
                norm_layer(nf_out),
            )
        layers = [block(
            nf_in, nf_out, stride,
            downsample=downsample,
            norm_layer=norm_layer
        )]
        for _ in range(1, num_blocks):
            layers.append(block(nf_out, nf_out, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, collect=False):
        x = maybe_cuda(x)

        # First component
        if len(self.precursors) == 0:
            h1 = self.layer0(x)
            h2 = self.layer1(h1)
            h3 = self.layer2(h2)
            h4 = self.layer3(h3)
            h5 = self.layer4(h4)
            h5 = F.avg_pool2d(h5, h5.size(2)).view(h5.size(0), -1)
            pred = self.predict(h5)

            if collect:
                return [pred], [
                    h1.detach(), h2.detach(), h3.detach(),
                    h4.detach(), h5.detach()]
            else:
                return pred

        # Second or layer component
        preds, features = self.precursors[-1](x, collect=True)
        h1 = self.layer0(x)
        h1_cat = torch.cat([features[0], h1], dim=1)
        h2 = self.layer1(h1_cat)
        h2_cat = torch.cat([features[1], h2], dim=1)
        h3 = self.layer2(h2_cat)
        h3_cat = torch.cat([features[2], h3], dim=1)
        h4 = self.layer3(h3_cat)
        h4_cat = torch.cat([features[3], h4], dim=1)
        h5 = self.layer4(h4_cat)
        h5 = F.avg_pool2d(h5, h5.size(2)).view(h5.size(0), -1)
        h5_cat = torch.cat([features[4], h5], dim=1)
        pred = self.predict(h5_cat)

        if collect:
            preds.append(pred)
            return preds, [
                h1_cat.detach(), h2_cat.detach(), h3_cat.detach(),
                h4_cat.detach(), h5_cat.detach(),
            ]
        else:
            return pred
