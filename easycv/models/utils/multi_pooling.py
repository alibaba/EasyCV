# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    """GemPooling used for image retrival
       p = 1, avgpooling
       p > 1 : increases the contrast of the pooled feature map and focuses on the salient features of the image
       p = infinite : spatial max-pooling layer
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeMPooling, self).__init__()
        # self.p = nn.Parameter(torch.ones(1)*p)
        self.p = p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p),
                            (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(
            self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class MultiPooling(nn.Module):
    """Pooling layers for features from multiple depth.
    """
    POOL_PARAMS = {
        'resnet50': [
            dict(kernel_size=10, stride=10, padding=4),
            dict(kernel_size=16, stride=8, padding=0),
            dict(kernel_size=13, stride=5, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=6, stride=1, padding=0)
        ]
    }
    POOL_SIZES = {'resnet50': [12, 6, 4, 3, 2]}
    POOL_DIMS = {'resnet50': [9216, 9216, 8192, 9216, 8192]}

    def __init__(self,
                 pool_type='adaptive',
                 in_indices=(0, ),
                 backbone='resnet50'):
        super(MultiPooling, self).__init__()
        assert pool_type in ['adaptive', 'specified']
        if pool_type == 'adaptive':
            self.pools = nn.ModuleList([
                nn.AdaptiveAvgPool2d(self.POOL_SIZES[backbone][l])
                for l in in_indices
            ])
        else:
            self.pools = nn.ModuleList([
                nn.AvgPool2d(**self.POOL_PARAMS[backbone][l])
                for l in in_indices
            ])

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        return [p(xx) for p, xx in zip(self.pools, x)]


class MultiAvgPooling(nn.Module):
    """Pooling layers for features from multiple depth.
    """
    POOL_PARAMS = {
        'resnet50': [
            dict(kernel_size=10, stride=10, padding=4),
            dict(kernel_size=16, stride=8, padding=0),
            dict(kernel_size=13, stride=5, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=7, stride=1, padding=0)
        ]
    }
    # POOL_SIZES = {'resnet50': [12, 6, 4, 3, 2]}
    POOL_SIZES = {'resnet50': [12, 6, 4, 3, 1]}
    POOL_DIMS = {'resnet50': [9216, 9216, 8192, 9216, 2048]}

    def __init__(self,
                 pool_type='adaptive',
                 in_indices=(0, ),
                 backbone='resnet50'):
        super(MultiAvgPooling, self).__init__()
        assert pool_type in ['adaptive', 'specified']
        if pool_type == 'adaptive':
            self.pools = nn.ModuleList([
                nn.AdaptiveAvgPool2d(self.POOL_SIZES[backbone][l])
                for l in in_indices
            ])
        else:
            self.pools = nn.ModuleList([
                nn.AvgPool2d(**self.POOL_PARAMS[backbone][l])
                for l in in_indices
            ])

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        return [p(xx) for p, xx in zip(self.pools, x)]
