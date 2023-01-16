# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from pathlib import Path

from PIL import Image
from scipy.io import loadmat
from torchvision.datasets.utils import (check_integrity,
                                        download_and_extract_archive,
                                        download_url)

from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class ClsSourceFlowers102(object):

    _download_url_prefix = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
    _file_dict = dict(  # filename, md5
        image=('102flowers.tgz', '52808999861908f626f3c1f4e79d11fa'),
        label=('imagelabels.mat', 'e0620be6f572b9609742df49c70aed4d'),
        setid=('setid.mat', 'a5357ecc9cb78c4bef273ce3793fc85c'))

    _splits_map = {'train': 'trnid', 'val': 'valid', 'test': 'tstid'}

    def __init__(self, root, split, download=False) -> None:

        assert split in ['train', 'test', 'val']
        self._base_folder = Path(root) / 'flowers-102'
        self._images_folder = self._base_folder / 'jpg'

        if download:
            self.download()
        # verify that the path exists
        if not self._check_integrity():
            raise FileNotFoundError(
                f'The data in the {self._base_folder} file directory is incomplete'
            )

        # Data reading in progress
        set_ids = loadmat(
            self._base_folder / self._file_dict['setid'][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[split]].tolist()

        labels = loadmat(
            self._base_folder / self._file_dict['label'][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels['labels'] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder /
                                     f'image_{image_id:05d}.jpg')

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):

        image_file, label = self._image_files[idx], self._labels[idx]
        img = Image.open(image_file).convert('RGB')
        result_dict = {'img': img, 'gt_labels': label}
        return result_dict

    # verify that the path exists
    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ['label', 'setid']:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        os.makedirs(self._base_folder, exist_ok=True)
        if self._check_integrity():
            return
        # Download and extract
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict['image'][1],
            remove_finished=True)
        for id in ['label', 'setid']:
            filename, md5 = self._file_dict[id]
            download_url(
                self._download_url_prefix + filename,
                str(self._base_folder),
                md5=md5)
