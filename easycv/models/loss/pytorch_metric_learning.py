# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect

import pytorch_metric_learning.losses as pml_losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from ..registry import LOSSES

# register all existing transforms in torchvision
for m in inspect.getmembers(pml_losses, inspect.isclass):
    LOSSES.register_module(m[1])


@LOSSES.register_module
class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self,
                 gamma=2,
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 num_classes=2):
        """
            FocalLoss2d, loss solve 2-class classification unbalance problem

            Args:
                gamma: focal loss param Gamma
                weight: weight same as loss._WeightedLoss
                size_average: size_average same as loss._WeightedLoss
                reduce : reduce same as loss._WeightedLoss
                reduction : reduce same as loss._WeightedLoss
                num_classes : fix num 2

            Returns:
                Focalloss nn.module.loss object
        """
        super(FocalLoss2d, self).__init__(weight, size_average, reduce,
                                          reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        """
            input: [N * num_classes]
            target : [N * num_classes] one-hot
        """

        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target)
        return loss


@LOSSES.register_module
class DistributeMSELoss(nn.Module):

    def __init__(self):
        """
            DistributeMSELoss : for faceid age, score predict (regression by softmax)
        """

        super(DistributeMSELoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, input, target):
        n, c = input.size()
        prob = self.softmax(input)

        distribute = torch.arange(0, c).repeat(n, 1).to(input.device)
        predict = (distribute * prob).sum(dim=1)
        return torch.mean(abs(predict - target))


@LOSSES.register_module
class CrossEntropyLossWithLabelSmooth(nn.Module):

    def __init__(self,
                 label_smooth=0.1,
                 temperature=1.0,
                 with_cls=False,
                 embedding_size=512,
                 num_classes=10000):
        """
        A softmax loss , with label_smooth and fc(to fit pytorch metric learning interface)
        Args:
            label_smooth: label_smooth args, default=0.1
            with_cls: if True, will generate a nn.Linear to trans input embedding from embedding_size to num_classes
            embedding_size : if input is feature not logits, then need this to indicate embedding shape
            num_classes : if input is feature not logits, then need this to indicate classification num_classes

        Returns:
            None
        Raises:
            IOError: An error occurred accessing the bigtable.Table object.
        """

        super(CrossEntropyLossWithLabelSmooth, self).__init__()
        self.label_smooth = label_smooth
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()
        self.temperature = temperature
        self.with_cls = with_cls

        if with_cls:
            self.cls = nn.Linear(embedding_size, num_classes, bias=False)
        else:
            self.cls = nn.Identity()

    def forward(self, input, target):

        if hasattr(self, 'cls') and self.with_cls:
            input = self.cls(input)
        target = target.long()
        input = input / self.temperature
        n, c = input.size()

        assert c > 1, 'No need for classification if c == 1'
        log_prob = self.log_softmax(input)
        loss = self.nllloss(log_prob, target)
        mean_logsum = torch.mean(torch.sum(log_prob, dim=1))
        return loss.mul(1 - self.label_smooth).sub(
            loss.add(mean_logsum).mul(self.label_smooth / (c - 1)))


@LOSSES.register_module
class AMSoftmaxLoss(nn.Module):

    def __init__(self,
                 embedding_size=512,
                 num_classes=100000,
                 margin=0.35,
                 scale=30):
        """
        AMsoftmax loss , with fc(to fit pytorch metric learning interface), paper: https://arxiv.org/pdf/1801.05599.pdf
        Args:

            embedding_size : forward input [N, embedding_size ]
            num_classes :  classification num_classes
            margin : AMSoftmax param
            scale : AMSoftmax param, should increase num_classes
        """
        super(AMSoftmaxLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.in_feats = embedding_size
        self.W = torch.nn.Parameter(
            torch.randn(embedding_size, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda:
            lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss


@LOSSES.register_module
class ModelParallelSoftmaxLoss(nn.Module):

    def __init__(self,
                 embedding_size=512,
                 num_classes=100000,
                 scale=None,
                 margin=None,
                 bias=True):
        """
        ModelParallel Softmax by sailfish
        Args:

            embedding_size : forward input [N, embedding_size ]
            num_classes :  classification num_classes
        """
        super(ModelParallelSoftmaxLoss, self).__init__()
        import sailfish
        rank, world_size = get_dist_info()
        self.model_parallel = sailfish.ModelParallel(rank, world_size)
        self.fc = sailfish.Linear(
            embedding_size,
            num_classes,
            bias=bias,
            weight_initializer=sailfish.ZerosInitializer(),
            bias_initializer=sailfish.OnesInitializer(),
            parallel=self.model_parallel)
        self.ce = sailfish.CrossEntropyLoss(parallel=self.model_parallel)

    def forward(self, x, lb):
        feature = self.model_parallel.gather(x)
        label = self.model_parallel.gather_target(lb)
        logits = self.fc(feature)
        loss = self.ce(logits, label)
        return loss


@LOSSES.register_module
class ModelParallelAMSoftmaxLoss(nn.Module):

    def __init__(self,
                 embedding_size=512,
                 num_classes=100000,
                 margin=0.35,
                 scale=30):
        """
        ModelParallel AMSoftmax by sailfish
        Args:

            embedding_size : forward input [N, embedding_size ]
            num_classes :  classification num_classes
        """
        super(ModelParallelAMSoftmaxLoss, self).__init__()

        import sailfish
        self.m = margin
        self.s = scale

        rank, world_size = get_dist_info()
        self.model_parallel = sailfish.ModelParallel(rank, world_size)

        self.fc = sailfish.AMLinear(
            embedding_size,
            num_classes,
            margin=self.m,
            scale=self.s,
            weight_initializer=sailfish.XavierUniformInitializer(),
            parallel=self.model_parallel)
        self.ce = sailfish.CrossEntropyLoss(parallel=self.model_parallel)

    def forward(self, x, lb):
        feature = self.model_parallel.gather(x)
        label = self.model_parallel.gather_target(lb)
        costh_m_s = self.fc(feature, label)
        # cosine = self.model_parallel.gather(cosine)
        # print(cosine.shape)

        # lb_view = label.view(-1, 1)
        # if lb_view.is_cuda: lb_view = lb_view.cpu()
        # delt_costh = torch.zeros(cosine.size()).scatter_(1, lb_view, self.m)
        # if features.is_cuda: delt_costh = delt_costh.cuda()
        # costh_m = cosine - delt_costh
        # costh_m_s = self.scale * costh_m
        loss = self.ce(costh_m_s, label)

        return loss


@LOSSES.register_module
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, num_classes=1000, **kwargs):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
