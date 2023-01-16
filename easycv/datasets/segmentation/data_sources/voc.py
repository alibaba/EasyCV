# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from multiprocessing import cpu_count
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive

from easycv.datasets.registry import DATASOURCES
from easycv.utils.constant import CACHE_DIR
from .raw import SegSourceRaw


@DATASOURCES.register_module
class SegSourceVoc2012(SegSourceRaw):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    data format is as follows:
    ```
    |- voc_data
        |-ImageSets
            |-Segmentation
                |-train.txt
                |-...
        |-JPEGImages
            |-00001.jpg
            |-...
        |-SegmentationClass
            |-00001.png
            |-...

    ```

    Args:
        download (bool): This parameter is optional. If download is True and path is not provided,
            a temporary directory is automatically created for downloading
        path (str): This parameter is optional. If download is True and path is not provided,
            a temporary directory is automatically created for downloading
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_root/label_root will be loaded.
        classes (str | list): classes list or file
        img_suffix (str): image file suffix
        label_suffix (str): label file suffix
        reduce_zero_label (bool): whether to mark label zero as ignored
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            palette of segmentation map, if none, random palette will be generated
        cache_at_init (bool): if set True, will cache in memory in __init__ for faster training
        cache_on_the_fly (bool): if set True, will cache in memroy during training
    """

    _download_url_ = {
        'url':
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': os.path.join('VOCdevkit', 'VOC2012')
    }

    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self,
                 download=False,
                 path=CACHE_DIR,
                 split=None,
                 reduce_zero_label=False,
                 palette=None,
                 num_processes=int(cpu_count() / 2),
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 **kwargs):

        if kwargs.get('cfg'):
            self._download_url_ = kwargs.get('cfg')

        self._base_folder = Path(path)
        self._file_folder = self._base_folder / self._download_url_['base_dir']
        if download:
            self.download()

        assert self._file_folder.exists(
        ), 'Dataset not found or corrupted. You can use download=True to download it'

        image_dir = self._file_folder / 'JPEGImages'
        mask_dir = self._file_folder / 'SegmentationClass'
        split_file = self._file_folder / self.split_file(split)

        if image_dir.exists() and mask_dir.exists() and split_file.exists():

            super(SegSourceVoc2012, self).__init__(
                img_root=str(image_dir),
                label_root=str(mask_dir),
                split=str(split_file),
                classes=self.VOC_CLASSES,
                img_suffix='.jpg',
                label_suffix='.png',
                reduce_zero_label=reduce_zero_label,
                palette=palette,
                num_processes=num_processes,
                cache_at_init=cache_at_init,
                cache_on_the_fly=cache_on_the_fly)

    def split_file(self, split):
        split_file = 'ImageSets/Segmentation'
        if split == 'train':
            split_file += '/train.txt'
        elif split == 'val':
            split_file += '/val.txt'
        else:
            split_file += '/trainval.txt'

        return split_file

    def download(self):
        if self._file_folder.exists():
            return

        # Download and extract
        download_and_extract_archive(
            self._download_url_.get('url'),
            str(self._base_folder),
            str(self._base_folder),
            md5=self._download_url_.get('md5'),
            remove_finished=True)


@DATASOURCES.register_module
class SegSourceVoc2010(SegSourceRaw):
    """Data source for semantic segmentation.
    data format is as follows:
    ```
    |- voc_data
        |-ImageSets
            |-Segmentation
                |-train.txt
                |-...
        |-JPEGImages
            |-00001.jpg
            |-...
        |-SegmentationClass
            |-00001.png
            |-...

    ```

    Args:
        download (bool): This parameter is optional. If download is True and path is not provided,
            a temporary directory is automatically created for downloading
        path (str): This parameter is optional. If download is True and path is not provided,
            a temporary directory is automatically created for downloading
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_root/label_root will be loaded.
        classes (str | list): classes list or file
        img_suffix (str): image file suffix
        label_suffix (str): label file suffix
        reduce_zero_label (bool): whether to mark label zero as ignored
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            palette of segmentation map, if none, random palette will be generated
        cache_at_init (bool): if set True, will cache in memory in __init__ for faster training
        cache_on_the_fly (bool): if set True, will cache in memroy during training
    """

    _download_url_ = {
        'url':
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': os.path.join('VOCdevkit', 'VOC2010')
    }

    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self,
                 download=False,
                 path=CACHE_DIR,
                 split=None,
                 reduce_zero_label=False,
                 palette=None,
                 num_processes=int(cpu_count() / 2),
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 **kwargs):

        if kwargs.get('cfg'):
            self._download_url_ = kwargs.get('cfg')

        self._base_folder = Path(path)
        self._file_folder = self._base_folder / self._download_url_['base_dir']
        if download:
            self.download()

        assert self._file_folder.exists(
        ), 'Dataset not found or corrupted. You can use download=True to download it'

        image_dir = self._file_folder / 'JPEGImages'
        mask_dir = self._file_folder / 'SegmentationClass'
        split_file = self._file_folder / self.split_file(split)

        if image_dir.exists() and mask_dir.exists() and split_file.exists():

            super(SegSourceVoc2010, self).__init__(
                img_root=str(image_dir),
                label_root=str(mask_dir),
                split=str(split_file),
                classes=self.VOC_CLASSES,
                img_suffix='.jpg',
                label_suffix='.png',
                reduce_zero_label=reduce_zero_label,
                palette=palette,
                num_processes=num_processes,
                cache_at_init=cache_at_init,
                cache_on_the_fly=cache_on_the_fly)

    def split_file(self, split):
        split_file = 'ImageSets/Segmentation'
        if split == 'train':
            split_file += '/train.txt'
        elif split == 'val':
            split_file += '/val.txt'
        else:
            split_file += '/trainval.txt'

        return split_file

    def download(self):
        if self._file_folder.exists():
            return self._file_folder

        # Download and extract
        download_and_extract_archive(
            self._download_url_.get('url'),
            str(self._base_folder),
            str(self._base_folder),
            md5=self._download_url_.get('md5'),
            remove_finished=True)


@DATASOURCES.register_module
class SegSourceVoc2007(SegSourceRaw):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    data format is as follows:
    ```
    |- voc_data
        |-ImageSets
            |-Segmentation
                |-train.txt
                |-...
        |-JPEGImages
            |-00001.jpg
            |-...
        |-SegmentationClass
            |-00001.png
            |-...

    ```

    Args:
        download (bool): This parameter is optional. If download is True and path is not provided,
            a temporary directory is automatically created for downloading
        path (str): This parameter is optional. If download is True and path is not provided,
            a temporary directory is automatically created for downloading
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_root/label_root will be loaded.
        classes (str | list): classes list or file
        img_suffix (str): image file suffix
        label_suffix (str): label file suffix
        reduce_zero_label (bool): whether to mark label zero as ignored
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            palette of segmentation map, if none, random palette will be generated
        cache_at_init (bool): if set True, will cache in memory in __init__ for faster training
        cache_on_the_fly (bool): if set True, will cache in memroy during training
    """

    _download_url_ = {
        'url':
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    }

    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self,
                 download=False,
                 path=CACHE_DIR,
                 split=None,
                 reduce_zero_label=False,
                 palette=None,
                 num_processes=int(cpu_count() / 2),
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 **kwargs):

        if kwargs.get('cfg'):
            self._download_url_ = kwargs.get('cfg')

        self._base_folder = Path(path)
        self._file_folder = self._base_folder / self._download_url_['base_dir']
        if download:
            self.download()

        assert self._file_folder.exists(
        ), 'Dataset not found or corrupted. You can use download=True to download it'

        image_dir = self._file_folder / 'JPEGImages'
        mask_dir = self._file_folder / 'SegmentationClass'
        split_file = self._file_folder / self.split_file(split)

        if image_dir.exists() and mask_dir.exists() and split_file.exists():

            super(SegSourceVoc2007, self).__init__(
                img_root=str(image_dir),
                label_root=str(mask_dir),
                split=str(split_file),
                classes=self.VOC_CLASSES,
                img_suffix='.jpg',
                label_suffix='.png',
                reduce_zero_label=reduce_zero_label,
                palette=palette,
                num_processes=num_processes,
                cache_at_init=cache_at_init,
                cache_on_the_fly=cache_on_the_fly)

    def split_file(self, split):
        split_file = 'ImageSets/Segmentation'
        if split == 'train':
            split_file += '/train.txt'
        elif split == 'val':
            split_file += '/val.txt'
        else:
            split_file += '/trainval.txt'

        return split_file

    def download(self):
        if self._file_folder.exists():
            return
        # Download and extract
        download_and_extract_archive(
            self._download_url_.get('url'),
            str(self._base_folder),
            str(self._base_folder),
            md5=self._download_url_.get('md5'),
            remove_finished=True)
