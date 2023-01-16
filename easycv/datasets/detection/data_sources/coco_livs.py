# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive
from xtcocotools.coco import COCO

from easycv.datasets.detection.data_sources.coco import DetSourceCoco
from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class DetSourceLvis(DetSourceCoco):
    """
    lvis data source
    """

    cfg = dict(
        links=[
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip',
            'http://images.cocodataset.org/zips/train2017.zip',
            'http://images.cocodataset.org/zips/val2017.zip'
        ],
        train='lvis_v1_train.json',
        val='lvis_v1_val.json',
        dataset='images'
        # default
    )

    def __init__(self,
                 pipeline,
                 path=None,
                 download=True,
                 split='train',
                 test_mode=False,
                 filter_empty_gt=False,
                 classes=None,
                 iscrowd=False,
                 **kwargs):
        """
        Args:
            path: This parameter is optional. If download is True and path is not provided,
                    a temporary directory is automatically created for downloading
            download: If the value is True, the file is automatically downloaded to the path directory.
                      If False, automatic download is not supported and data in the path is used
            split: train or val
            test_mode (bool, optional): If set True, `self._filter_imgs` will not works.
            filter_empty_gt (bool, optional): If set true, images without bounding
                boxes of the dataset's classes will be filtered out. This option
                only works when `test_mode=False`, i.e., we never filter images
                during tests.
            iscrowd: when traing setted as False, when val setted as True
        """
        if kwargs.get('cfg'):
            self.cfg = kwargs.get('cfg')

        assert split in ['train', 'val']
        assert os.path.isdir(path), f'{path} is not dir'
        self.lvis_path = Path(os.path.join(path, 'LVIS'))

        if download:
            self.download()

        else:
            if not (self.lvis_path.exists() and self.lvis_path.is_dir()):
                raise FileNotFoundError(
                    f'The data in the {self.lvis_path} file directory is not exists'
                )

        super(DetSourceLvis, self).__init__(
            ann_file=str(self.lvis_path / self.cfg.get(split)),
            img_prefix=str(self.lvis_path / self.cfg.get('dataset')),
            pipeline=pipeline,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            classes=classes,
            iscrowd=iscrowd)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.
        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = os.path.basename(info['coco_url'])
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def download(self):
        if not (self.lvis_path.exists() and self.lvis_path.is_dir()):
            for tmp_url in self.cfg.get('links'):
                download_and_extract_archive(
                    tmp_url,
                    self.lvis_path,
                    self.lvis_path,
                    remove_finished=True)
            self.merge_images_folder()
        return

    def merge_images_folder(self):
        new_images_folder = str(self.lvis_path / self.cfg.get('dataset'))
        os.rename(str(self.lvis_path / 'train2017'), new_images_folder)
        os.system(
            f"mv {str(self.lvis_path / 'val2017')}/* {new_images_folder} ")
        os.rmdir(str(self.lvis_path / 'val2017'))
