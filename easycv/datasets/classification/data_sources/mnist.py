# Copyright (c) Alibaba, Inc. and its affiliates.
from PIL import Image
from torchvision.datasets.mnist import MNIST

from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class ClsSourceMnist(MNIST):

    CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    resources = [
        ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
         'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
         'd53e105ee54ea40749a09fcbcd1e9432'),
        ('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
         '9fb629c4189551a2d022fa330f9573f3'),
        ('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
         'ec29112dd5afa0611ce80d1b7f02629c')
    ]

    def __init__(self, root, split, download=False):
        assert split in ['train', 'test']
        super(ClsSourceMnist, self).__init__(
            root=root, train=(split == 'train'), download=download)
        self.labels = self.targets

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict
