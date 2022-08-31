# Copyright (c) Alibaba, Inc. and its affiliates.
from PIL import Image
from torchvision.datasets.cifar import CIFAR10, CIFAR100

from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class ClsSourceCifar10(CIFAR10):

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self, root, split, download=False):
        assert split in ['train', 'test']
        super(ClsSourceCifar10, self).__init__(
            root=root, train=(split == 'train'), download=download)
        self.labels = self.targets

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]  # img: HWC, RGB
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict


@DATASOURCES.register_module
class ClsSourceCifar100(CIFAR100):

    CLASSES = None
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    def __init__(self, root, split, download=False):
        assert split in ['train', 'test']

        super(ClsSourceCifar100, self).__init__(
            root=root, train=(split == 'train'), download=download)

        self.labels = self.targets

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]  # img: HWC, RGB
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict
