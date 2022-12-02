# Copyright (c) OpenMMLab. All rights reserved.
import os

from tqdm import tqdm
from xtcocotools.coco import COCO

from easycv.datasets.registry import DATASOURCES
from .coco import DetSourceCoco


@DATASOURCES.register_module
class DetSourceObject365(DetSourceCoco):
    """
    Object 365 data source
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 pipeline,
                 test_mode=False,
                 filter_empty_gt=False,
                 classes=[],
                 iscrowd=False):
        """
        Args:
            ann_file: Path of annotation file.
            img_prefix: coco path prefix
            test_mode (bool, optional): If set True, `self._filter_imgs` will not works.
            filter_empty_gt (bool, optional): If set true, images without bounding
                boxes of the dataset's classes will be filtered out. This option
                only works when `test_mode=False`, i.e., we never filter images
                during tests.
            iscrowd: when traing setted as False, when val setted as True
        """

        super(DetSourceObject365, self).__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
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
        img_path = os.listdir(self.img_prefix)
        data_infos = []
        total_ann_ids = []
        for i in tqdm(self.img_ids, desc='Scaning Images'):
            info = self.coco.loadImgs([i])[0]
            filename = os.path.basename(info['file_name'])
            # Filter the information corresponding to the image
            if filename in img_path:
                info['filename'] = filename
                data_infos.append(info)
                ann_ids = self.coco.getAnnIds(imgIds=[i])
                total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        del total_ann_ids
        return data_infos
