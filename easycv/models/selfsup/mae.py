import torch

from .. import builder
from ..base import BaseModel
from ..registry import MODELS


@MODELS.register_module
class MAE(BaseModel):

    def __init__(self,
                 backbone,
                 neck,
                 mask_ratio=0.75,
                 norm_pix_loss=True,
                 **kwargs):
        super(MAE, self).__init__()
        assert backbone.get('patch_size', None) is not None, \
            'patch_size should be set'
        self.patch_size = backbone['patch_size']
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.encoder = builder.build_backbone(backbone)
        neck['num_patches'] = self.encoder.num_patches
        self.decoder = builder.build_neck(neck)

    def patchify(self, imgs):
        """convert image to patch

        Args:
            imgs: (N, 3, H, W)
        Returns:
            x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_loss(self, imgs, pred, mask):
        """compute loss

        Args:
            imgs: (N, 3, H, W)
            pred: (N, L, p*p*3)
            mask: (N, L), 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_train(self, img, **kwargs):
        latent, mask, ids_restore = self.encoder(
            img, mask_ratio=self.mask_ratio)
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(img, pred, mask)
        return dict(loss=loss)

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        else:
            raise Exception('No such mode: {}'.format(mode))
