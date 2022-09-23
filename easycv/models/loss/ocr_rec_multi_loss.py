# Modified from https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/ppocr/losses
import torch
from torch import nn

from easycv.models.builder import LOSSES


@LOSSES.register_module()
class CTCLoss(nn.Module):

    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, labels, label_lengths):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        # predicts = predicts.transpose(1, 0, 2)
        predicts = predicts.permute(1, 0, 2).contiguous()
        predicts = predicts.log_softmax(2)
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor([N] * B, dtype=torch.int32)
        labels = labels.type(torch.int32)
        label_lengths = label_lengths.type(torch.int64)

        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = torch.subtract(torch.tensor([1.0]), weight)
            weight = torch.square(weight)
            loss = torch.multiply(loss, weight)
        loss = loss.mean()
        return {'loss': loss}


@LOSSES.register_module()
class SARLoss(nn.Module):

    def __init__(self, **kwargs):
        super(SARLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 92)  # 6626
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='mean', ignore_index=ignore_index)

    def forward(self, predicts, label):
        predict = predicts[:, :
                           -1, :]  # ignore last index of outputs to be in same seq_len with targets
        label = label.type(
            torch.int64
        )[:, 1:]  # ignore first index of target in loss calculation
        batch_size, num_steps, num_classes = predict.shape[0], predict.shape[
            1], predict.shape[2]
        assert len(label.shape) == len(list(predict.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = torch.reshape(predict, [-1, num_classes])
        targets = torch.reshape(label, [-1])
        loss = self.loss_func(inputs, targets)
        return {'loss': loss}


@LOSSES.register_module()
class MultiLoss(nn.Module):

    def __init__(self,
                 loss_config_list,
                 weight_1=1.0,
                 weight_2=1.0,
                 gtc_loss='sar',
                 **kwargs):
        super().__init__()
        self.loss_funcs = {}
        self.loss_list = loss_config_list
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.gtc_loss = gtc_loss
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss

    def forward(self, predicts, label_ctc=None, label_sar=None, length=None):
        self.total_loss = {}
        total_loss = 0.0
        # batch [image, label_ctc, label_sar, length, valid_ratio]
        for name, loss_func in self.loss_funcs.items():
            if name == 'CTCLoss':
                loss = loss_func(predicts['ctc'], label_ctc,
                                 length)['loss'] * self.weight_1
            elif name == 'SARLoss':
                loss = loss_func(predicts['sar'],
                                 label_sar)['loss'] * self.weight_2
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiLoss yet'.format(name))
            self.total_loss[name] = loss
            total_loss += loss
        self.total_loss['loss'] = total_loss
        return self.total_loss
