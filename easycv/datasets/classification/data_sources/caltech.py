# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from torchvision.datasets import Caltech101, Caltech256
from torchvision.datasets.utils import (download_and_extract_archive,
                                        extract_archive)

from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class ClsSourceCaltech101(object):

    def __init__(self, root, download=True):

        if download:
            root = self.download(root)
            self.caltech101 = Caltech101(root, 'category', download=False)
        else:
            self.caltech101 = Caltech101(root, 'category', download=False)

        # data label_classes
        self.CLASSES = self.caltech101.categories

    def __len__(self):
        return len(self.caltech101.index)

    def __getitem__(self, idx):
        # img: HWC, RGB
        img, label = self.caltech101[idx]
        print(self.caltech101[idx])
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict

    def download(self, root):

        if os.path.exists(os.path.join(root, 'caltech101')):
            return root

        if os.path.exists(os.path.join(root, 'caltech-101.zip')):
            self.downloaded_exists(root)
            return root

        # download and extract the file
        download_and_extract_archive(
            'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1',
            root,
            filename='caltech-101.zip',
            md5='3138e1922a9193bfa496528edbbc45d0',
            remove_finished=True)
        self.normalized_path(root)

        return root

    # The data has been downloaded and decompressed
    def downloaded_exists(self, root):
        extract_archive(
            os.path.join(root, 'caltech-101.zip'), root, remove_finished=True)
        self.normalized_path(root)

    # The routinized path meets the input requirements
    def normalized_path(self, root):
        # rename root path
        old_folder_name = os.path.join(root, 'caltech-101')
        new_folder_name = os.path.join(root, 'caltech101')
        os.rename(old_folder_name, new_folder_name)
        # extract object file
        img_categories = os.path.join(new_folder_name,
                                      '101_ObjectCategories.tar.gz')
        extract_archive(img_categories, new_folder_name, remove_finished=True)


@DATASOURCES.register_module
class ClsSourceCaltech256(object):

    def __init__(self, root, download=True):

        if download:
            self.download(root)
            self.caltech256 = Caltech256(root, download=False)
        else:
            self.caltech256 = Caltech256(root, download=False)

        # data classes
        self.CLASSES = self.caltech256.categories

    def __len__(self):
        return len(self.caltech256.index)

    def __getitem__(self, idx):
        # img: HWC, RGB
        img, label = self.caltech256[idx]
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict

    def download(self, root):

        caltech256_path = os.path.join(root, 'caltech256')

        if os.path.exists(caltech256_path):
            return

        download_and_extract_archive(
            'https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1',
            caltech256_path,
            filename='256_ObjectCategories.tar',
            md5='67b4f42ca05d46448c6bb8ecd2220f6d',
            remove_finished=True)
