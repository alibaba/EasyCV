# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F
from torch import nn

from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.utils.checkpoint import get_checkpoint


# Modified from https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/recognizers/recognizer3d.py
@MODELS.register_module()
class Recognizer3D(BaseModel):
    """3D recognizer model.
    """

    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Recognizer3D, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck) if neck else None
        self.cls_head = builder.build_head(cls_head) if cls_head else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = get_checkpoint(
            pretrained) if pretrained else pretrained
        self.activate_fn = nn.Softmax(dim=1)

        # aux_info is the list of tensor names beyond 'imgs' and 'label' which
        # will be used in train_step and val_step, data_batch should contain
        # these tensors
        self.aux_info = []
        if train_cfg is not None and 'aux_info' in train_cfg:
            self.aux_info = train_cfg['aux_info']
        # max_testing_views should be int
        self.max_testing_views = None
        if test_cfg is not None and 'max_testing_views' in test_cfg:
            self.max_testing_views = test_cfg['max_testing_views']
            assert isinstance(self.max_testing_views, int)

        if test_cfg is not None and 'feature_extraction' in test_cfg:
            self.feature_extraction = test_cfg['feature_extraction']
        else:
            self.feature_extraction = False

        # mini-batch blending, e.g. mixup, cutmix, etc.
        # self.blending = None
        # if train_cfg is not None and 'blending' in train_cfg:
        #     from mmcv.utils import build_from_cfg
        #     from mmaction.datasets.builder import BLENDINGS
        #     self.blending = build_from_cfg(train_cfg['blending'], BLENDINGS)

        self.init_weights()

    @property
    def with_neck(self):
        """bool: whether the recognizer has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        """Initialize the model network weights."""
        if isinstance(self.pretrained, str):
            self.backbone.init_weights(pretrained=self.pretrained)
        else:
            self.backbone.init_weights()

        if self.with_cls_head and hasattr(self.cls_head, 'init_weights'):
            self.cls_head.init_weights()
        if self.with_neck and hasattr(self.neck, 'init_weights'):
            self.neck.init_weights()

    def average_clip(self, cls_score, num_segs=1):
        """Averaging class score over multiple clips.
        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.
        Args:
            cls_score (torch.Tensor): Class score to be averaged.
            num_segs (int): Number of clips for each input sample.
        Returns:
            torch.Tensor: Averaged class score.
        """
        if 'average_clips' not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips is None:
            return cls_score

        batch_size = cls_score.shape[0]
        cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=1)

        return cls_score

    def extract_feat(self, imgs):
        """Extract features through a backbone.
        Args:
            imgs (torch.Tensor): The input images.
        Returns:
            torch.tensor: The extracted features.
        """

        x = self.backbone(imgs)
        return x

    def forward_train(self, imgs, label, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
            x, loss_aux = self.neck(x, label.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = label.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            # perform spatio-temporal pooling
            avg_pool = nn.AdaptiveAvgPool3d(1)
            if isinstance(feat, tuple):
                feat = [avg_pool(x) for x in feat]
                # concat them
                feat = torch.cat(feat, axis=1)
            else:
                feat = avg_pool(feat)
            # squeeze dimensions
            feat = feat.reshape((batches, num_segs, -1))
            # temporal average pooling
            feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs, label=None, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        cls_score = self._do_test(imgs)
        if label is not None:
            return dict(neck=cls_score.cpu(), label=label.cpu())
        else:
            result = {}
            result['prob'] = self.activate_fn(cls_score.cpu())
            if 'img_metas' in kwargs and 'filename' in kwargs['img_metas'][0]:
                result['filename'] = [kwargs['img_metas'][0]['filename']]
            # result['class'] = torch.argmax(result['prob'])
            # print(result['class'])
            return result

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the \
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data, mode='train')
        loss, log_vars = self._parse_losses(losses)

        if type(data['imgs']) == list:
            num_samples = len(data['imgs'][0])
        else:
            num_samples = len(data['imgs'].data)

        return dict(loss=loss, log_vars=log_vars, num_samples=num_samples)
