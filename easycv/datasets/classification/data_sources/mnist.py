# Copyright (c) Alibaba, Inc. and its affiliates.
from PIL import Image
from torchvision.datasets import MNIST, FashionMNIST

from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class ClsSourceMnist(object):

    def __init__(self, root, split, download=True):
        assert split in ['train', 'test']
        self.mnist = MNIST(
            root=root, train=(split == 'train'), download=download)
        self.labels = self.mnist.targets
        # data label_classes
        self.CLASSES = self.mnist.classes

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # img: HWC, RGB
        img = Image.fromarray(self.mnist.data[idx].numpy()).convert('RGB')
        label = self.labels[idx].item()
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict


@DATASOURCES.register_module
class ClsSourceFashionMnist(object):

    def __init__(self, root, split, download=True):
        assert split in ['train', 'test']
        self.fashion_mnist = FashionMNIST(
            root=root, train=(split == 'train'), download=download)
        self.labels = self.fashion_mnist.targets
        # data label_classes
        self.CLASSES = self.fashion_mnist.classes

    def __len__(self):
        return len(self.fashion_mnist)

    def __getitem__(self, idx):
        # img: HWC, RGB
        img = Image.fromarray(
            self.fashion_mnist.data[idx].numpy()).convert('RGB')
        label = self.labels[idx].item()
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict
