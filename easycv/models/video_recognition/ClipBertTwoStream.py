# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.utils.checkpoint import get_checkpoint


@MODELS.register_module()
class ClipBertTwoStream(BaseModel):

    def __init__(self,
                 vision,
                 text,
                 train_cfg=None,
                 test_cfg=None,
                 vison_pretrained=None,
                 text_pretrained=None,
                 multi_class=False):
        super(ClipBertTwoStream, self).__init__()
        self.vision = builder.build_backbone(vision)
        self.text = builder.build_backbone(text)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.multi_class = multi_class

        self.vison_pretrained = get_checkpoint(
            vison_pretrained) if vison_pretrained else vison_pretrained
        self.text_pretrained = get_checkpoint(
            text_pretrained) if text_pretrained else text_pretrained
        loss_cls = dict(type='CrossEntropyLoss') if not multi_class else dict(
            type='AsymmetricLoss')
        self.loss_cls = builder.build_loss(loss_cls)
        self.activate_fn = nn.Softmax(
            dim=1) if not multi_class else nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        self.vision.init_weights(pretrained=self.vison_pretrained)
        self.text.init_weights(pretrained=self.text_pretrained)

    def extract_feat(self, imgs, text_input_ids, text_input_mask):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        visual_feature = self.vision(imgs)
        visual_feature = rearrange(visual_feature, 'n c d h w -> n d h w c')
        # visual_feature = torch.mean(visual_feature,1,True)
        logits = self.text(
            text_input_ids=text_input_ids,
            visual_inputs=visual_feature,
            text_input_mask=text_input_mask,
        )

        return logits

    def forward_train(self, imgs, text_input_ids, text_input_mask, label,
                      **kwargs):
        """Defines the computation performed at every call when training."""
        losses = dict()
        cls_score = self.extract_feat(imgs, text_input_ids, text_input_mask)

        gt_labels = torch.flatten(label) if label.shape[-1] == 1 else label
        loss_cls = self.loss_cls(cls_score, gt_labels)
        losses['loss_cls'] = loss_cls

        return losses

    def forward_test(self,
                     imgs,
                     text_input_ids,
                     text_input_mask,
                     label=None,
                     **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        cls_score = self.extract_feat(imgs, text_input_ids, text_input_mask)
        if label is not None:
            return dict(neck=cls_score.cpu(), label=label.cpu())
        else:
            result = {}
            result['prob'] = self.activate_fn(cls_score.cpu())
            if 'img_metas' in kwargs and 'filename' in kwargs['img_metas'][0]:
                result['filename'] = [kwargs['img_metas'][0]['filename']]
            # if not self.multi_class:
            #     result['class'] = torch.argmax(result['prob'])
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
