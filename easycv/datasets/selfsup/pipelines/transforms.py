# Copyright (c) Alibaba, Inc. and its affiliates.
from distutils.version import LooseVersion

import cv2
import numpy as np
import torch
from PIL import Image
from timm.data import create_transform
from torchvision import transforms as _transforms

from easycv.datasets.registry import PIPELINES
from easycv.utils.registry import build_from_cfg


@PIPELINES.register_module
class MAEFtAugment(object):
    """RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation
    with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    This code is borrowed from <https://github.com/pengzhiliang/MAE-pytorch>
    Args:
        input_size(int): images input size
        color_jitter(float): Color jitter factor
        auto_augment: Use AutoAugment policy
        iterpolation: Training interpolation
        re_prob: Random erase prob
        re_mode: Random erase mode
        re_count: Random erase count
        mean: mean used for normalization
        std: std used for normalization
        is_train: If True use all augmentation strategy
    """

    def __init__(self,
                 input_size=None,
                 color_jitter=None,
                 auto_augment=None,
                 interpolation=None,
                 re_prob=None,
                 re_mode=None,
                 re_count=None,
                 mean=None,
                 std=None,
                 is_train=True):
        resize_im = input_size > 32
        if is_train:
            self.trans = create_transform(
                input_size=input_size,
                is_training=True,
                color_jitter=color_jitter,
                auto_augment=auto_augment,
                interpolation=interpolation,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                mean=mean,
                std=std,
            )
        else:
            t = []
            if resize_im:
                if input_size < 384:
                    crop_pct = 224 / 256
                else:
                    crop_pct = 1.0
                size = int(input_size / crop_pct)
                t.append(
                    _transforms.Resize(
                        size, interpolation=3
                    ),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(_transforms.CenterCrop(input_size))
            t.append(_transforms.ToTensor())
            t.append(_transforms.Normalize(mean, std))
            self.trans = _transforms.Compose(t)

    def __call__(self, results):
        img = results['img']
        img = self.trans(img)
        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


class _RandomApply(_transforms.RandomApply):

    def forward(self, results):
        if self.p < torch.rand(1):
            return results
        for t in self.transforms:
            results = t(results)
        return results


@PIPELINES.register_module
class RandomAppliedTrans(object):
    '''Randomly applied transformations.
    Args:
        transforms (List[Dict]): List of transformations in dictionaries.
    '''

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _RandomApply(t, p=p)

    def __call__(self, results):
        return self.trans(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


# custom transforms
@PIPELINES.register_module
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""
    _IMAGENET_PCA = {
        'eigval':
        torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
        torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            assert isinstance(img, torch.Tensor), \
                'Expect torch.Tensor, got {}'.format(type(img))
            if self.alphastd == 0:
                continue

            alpha = img.new().resize_(3).normal_(0, self.alphastd)
            rgb = self.eigvec.type_as(img).clone()\
                .mul(alpha.view(1, 3).expand(3, 3))\
                .mul(self.eigval.view(1, 3).expand(3, 3))\
                .sum(1).squeeze()

            results[key] = img.add(rgb.view(3, 1, 1).expand_as(img))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


if LooseVersion(torch.__version__) < LooseVersion('1.7.0'):

    @PIPELINES.register_module
    class GaussianBlur(object):

        def __init__(self, kernel_size, sigma=(0.1, 2.0)):
            self.sigma_min = sigma[0]
            self.sigma_max = sigma[1]
            self.kernel_size = kernel_size

        def __call__(self, results):
            for key in results.get('img_fields', ['img']):
                img = results[key]
                sigma = np.random.uniform(self.sigma_min, self.sigma_max)
                img = cv2.GaussianBlur(
                    np.array(img), (self.kernel_size, self.kernel_size), sigma)
                results[key] = Image.fromarray(img.astype(np.uint8))

            return results

        def __repr__(self):
            repr_str = self.__class__.__name__
            return repr_str


@PIPELINES.register_module
class Solarization(object):

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = np.array(img)
            img = np.where(img < self.threshold, img, 255 - img)
            results[key] = Image.fromarray(img.astype(np.uint8))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
