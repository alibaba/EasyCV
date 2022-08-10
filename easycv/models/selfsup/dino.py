# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from easycv.utils.preprocess_function import (gaussianBlurDynamic,
                                              randomGrayScale, solarize)
from .. import builder
from ..base import BaseModel
from ..registry import MODELS


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple) or isinstance(_out, list):
                _out = _out[0]

            # some backbone doesn't contains avgpool
            if len(_out.size()) > 2:
                bs = _out.size(0)
                _out = self.avgpool(_out).view(bs, -1)

            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx

        # Run the head forward on the concatenated features.
        # return self.head(output)
        tp = self.head(output)
        return tp


class DINOLoss(nn.Module):

    def __init__(self,
                 out_dim,
                 ncrops,
                 warmup_teacher_temp,
                 teacher_temp,
                 warmup_teacher_temp_epochs,
                 nepochs,
                 device,
                 student_temp=0.1,
                 center_momentum=0.9):

        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        center = torch.zeros(1, out_dim).to(device)
        self.register_buffer('center', center)
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (np.linspace(warmup_teacher_temp, teacher_temp,
                         warmup_teacher_temp_epochs),
             np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]

        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(
                    -q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)

        _, world_size = get_dist_info()
        if world_size > 1:
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * world_size)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum)


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith('.bias') or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{
        'params': regularized
    }, {
        'params': not_regularized,
        'weight_decay': 0.
    }]


@MODELS.register_module
class DINO(BaseModel):

    def __init__(self,
                 backbone,
                 train_preprocess=[],
                 neck=None,
                 config=None,
                 pretrained=None):
        """ Init Moby

        Args:
            backbone: backbone config to build vision backbone
            train_preprocess: [gaussBlur, mixUp, solarize]
            neck: neck config to build Moby Neck
            config: DINO parameter config
        """
        super(DINO, self).__init__()

        self.config = config
        self.train_preprocess = train_preprocess  # we dont need it

        self.use_tfrecord_input = self.config.get('use_tfrecord_input', False)
        # dino has 3 augment pipeline, if use_tfrecord_input == True, use this
        self.train_preprocess_t1 = {
            'randomGrayScale': {
                'apply_prob': 0.2
            },
            'gaussianBlurDynamic': {
                'apply_prob': 1.0
            }
        }
        self.train_preprocess_t2 = {
            'randomGrayScale': {
                'apply_prob': 0.2
            },
            'gaussianBlurDynamic': {
                'apply_prob': 0.1
            },
            'solarize': {
                'threshold': 0.5,
                'apply_prob': 0.2
            }
        }
        self.train_preprocess_s = {
            'randomGrayScale': {
                'apply_prob': 0.2
            },
            'gaussianBlurDynamic': {
                'apply_prob': 0.5
            }
        }
        self.preprocess_key_map = {
            'randomGrayScale': randomGrayScale,
            'gaussianBlurDynamic': gaussianBlurDynamic,
            'solarize': solarize
        }

        # build model backbone
        teacher = builder.build_backbone(backbone)
        # vit based model, students should assign drop_path_rate
        if backbone.get('drop_path_rate', None) is not None:
            print('drop_path_rate : ', self.config['drop_path_rate'])
            backbone['drop_path_rate'] = self.config['drop_path_rate']
        student = builder.build_backbone(backbone)
        self.backbone = student

        # build model neck
        self.tneck = builder.build_neck(neck)
        neck['use_bn'] = config['use_bn_in_head']
        neck['norm_last_layer'] = config['norm_last_layer']
        sneck = builder.build_neck(neck)

        # combine with multi-crop infer wrapper
        self.teacher = MultiCropWrapper(teacher, sneck)
        self.student = MultiCropWrapper(student, self.tneck)

        if has_batchnorms(student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.teacher)

        self.init_no_ddp_teacher = False
        self.cur_epoch = 0
        # self.optim_param = get_params_groups(self.student)

    def get_params_groups(self):
        model = self.student
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith('.bias') or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{
            'params': regularized
        }, {
            'params': not_regularized,
            'weight_decay': 0.
        }]

    def init_weights(self, pretrained=None):
        # TODO: unify the use of init_weight
        raise ValueError('Dino `init_weights` has done in backbone and neck')
        # if pretrained is not None:
        #     print_log('load model from: {}'.format(pretrained), logger='root')
        # self.backbone.init_weights(pretrained=pretrained)
        # self.neck.init_weights(init_linear='kaiming')

    def init_before_train(self):
        # assign teacher model
        if hasattr(self.teacher, 'module'):
            self.teacher_without_ddp = self.teacher.module
        else:
            self.teacher_without_ddp = self.teacher

        self.teacher_without_ddp.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.dino_loss = DINOLoss(
            out_dim=self.config.get('out_dim', 65536),
            ncrops=int(self.config.get('local_crops_number', 8)) + 2,
            warmup_teacher_temp=self.config.get('warmup_teacher_temp', 0.04),
            teacher_temp=self.config.get('teacher_temp', 0.4),
            warmup_teacher_temp_epochs=self.config.get(
                'warmup_teacher_temp_epochs', 10),
            nepochs=self.config.get('epochs', 100),
            device=p.device)

        return

    def momentum_update_key_encoder(self, m=0.999):
        """ ema for dino
        """
        with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.student.parameters(),
                                        self.teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def forward_train(self, inputs):
        # move this part to DINOHOOK
        # if not self.init_no_ddp_teacher:
        #     self.init_before_train()
        #     self.init_no_ddp_teacher = True
        # else:
        #     with torch.no_grad():  # no gradient to keys
        #         self.momentum_update_key_encoder()

        self.student.train()

        # tensor operate to support some data aug
        input_list = []

        if self.use_tfrecord_input:  # DINO DATAAUG SET
            #  data aug for view 0(teacher1)
            img = inputs[0]
            for k in self.train_preprocess_t1.keys():
                img = self.preprocess_key_map[k](img,
                                                 **self.train_preprocess_t1[k])
            input_list.append(img)

            #  data aug for view 1(teacher2)
            img = inputs[1]
            for k in self.train_preprocess_t2.keys():
                img = self.preprocess_key_map[k](img,
                                                 **self.train_preprocess_t2[k])
            input_list.append(img)

            #  data aug for view 2~x(students)
            for img in inputs[2:]:
                for k in self.train_preprocess_s.keys():
                    img = self.preprocess_key_map[k](
                        img, **self.train_preprocess_s[k])
                input_list.append(img)
        else:
            input_list = inputs

        # normalize the prototypes
        teacher_output = self.teacher(
            input_list[:2])  # only the 2 global views pass through the teacher
        student_output = self.student(input_list)
        loss = self.dino_loss(student_output, teacher_output, self.cur_epoch)

        if not math.isfinite(loss.item()):
            print(
                'Loss is {}, stopping training'.format(loss.item()),
                force=True)
            sys.exit(1)

        if hasattr(self, 'this_loss'):
            self.this_loss = loss
            self.count = 1

        losses = dict()
        losses['loss'] = loss
        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward_feature(self, img, **kwargs):
        """Forward backbone

        Returns:
            x (torch.Tensor): feature tensor
        """
        # TODO: fix extract feature
        return_dict = {}
        x = self.student(img)
        return_dict['backbone'] = x

        if hasattr(self, 'tneck') and self.tneck is not None:
            feature = self.tneck([self.avg_pool(i) for i in x])[0]
        else:
            feature = self.avg_pool(x[-1])
        return_dict['neck'] = feature

        return return_dict

    def forward(self,
                img,
                gt_label=None,
                mode='train',
                extract_list=['neck'],
                **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)

        elif mode == 'extract':
            raise NotImplementedError()
            # rd = self.forward_feature(img)
            # rv = {}
            # for name in extract_list:
            #     if name in rd.keys():
            #         rv[name] = rd[name]
            #     else:
            #         raise 'Extract %s is not support in classification models' % name
            # if gt_label is not None:
            #     rv['gt_labels'] = gt_label.cpu()
            # return rv
        else:
            raise Exception('No such mode: {}'.format(mode))
