# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger
from easycv.utils.preprocess_function import gaussianBlur, randomGrayScale
from .. import builder
from ..base import BaseModel
from ..registry import MODELS
from ..utils import GatherLayer


@MODELS.register_module
class SimCLR(BaseModel):

    def __init__(self,
                 backbone,
                 train_preprocess=[],
                 neck=None,
                 head=None,
                 pretrained=None):
        super(SimCLR, self).__init__()
        self.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)

        self.preprocess_key_map = {
            'randomGrayScale': randomGrayScale,
            'gaussianBlur': gaussianBlur
        }
        self.train_preprocess = [
            self.preprocess_key_map[i] for i in train_preprocess
        ]
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.init_weights()

    @staticmethod
    def _create_buffer(N):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self.backbone, self.pretrained, strict=False, logger=logger)
        else:
            self.backbone.init_weights()
        self.neck.init_weights(init_linear='kaiming')

    def forward_backbone(self, img):
        """Forward backbone

        Returns:
            x (tuple): backbone outputs
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, **kwargs):
        assert isinstance(img, list)
        img = torch.stack(img, 1)
        img = img.reshape(
            img.size(0) * 2, img.size(2), img.size(3), img.size(4))

        for preprocess in self.train_preprocess:
            img = preprocess(img)
        x = self.forward_backbone(img)  # 2n
        z = self.neck(x)[0]  # (2n)xd
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)
        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception('No such mode: {}'.format(mode))
