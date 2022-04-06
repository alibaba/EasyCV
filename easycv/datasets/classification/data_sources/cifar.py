# Copyright (c) Alibaba, Inc. and its affiliates.
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100

from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class ClsSourceCifar10(object):

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]

    def __init__(self, root, split):
        assert split in ['train', 'test']
        self.cifar = CIFAR10(
            root=root, train=(split == 'train'), download=False)
        self.labels = self.cifar.targets

    def get_length(self):
        return len(self.cifar)

    def get_sample(self, idx):
        img = Image.fromarray(self.cifar.data[idx])
        label = self.labels[idx]  # img: HWC, RGB
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict


@DATASOURCES.register_module
class ClsSourceCifar100(object):

    CLASSES = None

    def __init__(self, root, split):
        assert split in ['train', 'test']

        self.cifar = CIFAR100(
            root=root, train=(split == 'train'), download=False)

        self.labels = self.cifar.targets

    def get_length(self):
        return len(self.cifar)

    def get_sample(self, idx):
        img = Image.fromarray(self.cifar.data[idx])
        label = self.labels[idx]  # img: HWC, RGB
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict
