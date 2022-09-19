import os
from collections import defaultdict

import mmcv
import numpy as np
from xtcocotools.coco import COCO

from easycv.datasets.detection.data_sources import DetSourceCoco
from easycv.datasets.registry import DATASOURCES, PIPELINES
from easycv.datasets.shared.pipelines import Compose
from easycv.framework.errors import RuntimeError, TypeError
from easycv.utils.registry import build_from_cfg

try:
    import panopticapi
    from panopticapi.evaluation import VOID
    from panopticapi.utils import id2rgb
except ImportError:
    panopticapi = None
    id2rgb = None
    VOID = None

INSTANCE_OFFSET = 1000


class COCOPanoptic(COCO):
    """This wrapper is for loading the panoptic style annotation file.

    The format is shown in the CocoPanopticDataset class.

    Args:
        annotation_file (str): Path of annotation file.
    """

    def __init__(self, annotation_file=None):
        if panopticapi is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super(COCOPanoptic, self).__init__(annotation_file)

    def createIndex(self):
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann, img_info in zip(self.dataset['annotations'],
                                     self.dataset['images']):
                img_info['segm_file'] = ann['file_name']
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    seg_ann['height'] = img_info['height']
                    seg_ann['width'] = img_info['width']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def load_anns(self, ids=[]):
        """Load anns with the specified ids.

        self.anns is a list of annotation lists instead of a
        list of annotations.

        Args:
            ids (int array): integer ids specifying anns

        Returns:
            anns (object array): loaded ann objects
        """
        anns = []

        if hasattr(ids, '__iter__') and hasattr(ids, '__len__'):
            # self.anns is a list of annotation lists instead of
            # a list of annotations
            for id in ids:
                anns += self.anns[id]
            return anns
        elif type(ids) == int:
            return self.anns[ids]


@DATASOURCES.register_module
class DetSourceCocoPanoptic(DetSourceCoco):
    """
    cocopanoptic data source
    """

    def __init__(self,
                 ann_file,
                 pan_ann_file,
                 img_prefix,
                 seg_prefix,
                 pipeline,
                 outfile_prefix='test/test_pan',
                 test_mode=False,
                 filter_empty_gt=False,
                 thing_classes=None,
                 stuff_classes=None,
                 iscrowd=False):
        """

        Args:
            ann_file (str): Path of coco detection annotation file
            pan_ann_file (str): Path of coco panoptic annotation file
            img_prefix (str): Path of image file
            seg_prefix (str): Path of semantic image file
            pipeline (list[dict]): list of data augmentatin operation
            outfile_prefix (str, optional): The filename prefix of the output files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.panoptic.json", "somepath/xxx.bbox.json",
                "somepath/xxx.segm.json"
            test_mode (bool, optional): If set True, `self._filter_imgs` will not works.
            filter_empty_gt (bool, optional): If set true, images without bounding
                boxes of the dataset's classes will be filtered out. This option
                only works when `test_mode=False`, i.e., we never filter images
                during tests.
            thing_classes (list[str], optional): list of thing classes. Defaults to None.
            stuff_classes (list[str], optional): list of thing classes. Defaults to None.
            iscrowd (bool, optional): when traing setted as False, when val setted as True. Defaults to False.
        """
        super().__init__(
            ann_file,
            img_prefix,
            pipeline,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            classes=thing_classes,
            iscrowd=iscrowd)
        self.outfile_prefix = outfile_prefix
        self.pan_ann_file = pan_ann_file
        self.seg_prefix = seg_prefix
        self.thing_classes = thing_classes
        self.stuff_classes = stuff_classes
        # load annotations (and proposals)
        self.data_infos_pan = self.load_annotations_pan(self.pan_ann_file)
        if not test_mode:
            valid_inds = self._filter_imgs_pan()
            self.data_infos_pan = [self.data_infos_pan[i] for i in valid_inds]
            self._set_group_flag_pan()
        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                transforms.append(transform)
            elif callable(transform):
                transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')
        self.pipeline = Compose(transforms)

    def load_annotations_pan(self, ann_file):
        """Load annotation from COCO Panoptic style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco_pan = COCOPanoptic(ann_file)
        self.cat_ids_pan = self.coco_pan.getCatIds()
        self.cat2label_pan = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids_pan)
        }
        self.categories_pan = self.coco_pan.cats
        self.img_ids_pan = self.coco_pan.getImgIds()
        data_infos = []
        for i in self.img_ids_pan:
            info = self.coco_pan.loadImgs([i])[0]
            info['filename'] = info['file_name']
            info['segm_file'] = info['filename'].replace('jpg', 'png')
            data_infos.append(info)
        return data_infos

    def get_ann_info_pan(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos_pan[idx]['id']
        ann_ids = self.coco_pan.getAnnIds(imgIds=[img_id])
        ann_info = self.coco_pan.load_anns(ann_ids)
        # filter out unmatched images
        ann_info = [i for i in ann_info if i['image_id'] == img_id]
        return self._parse_ann_info_pan(self.data_infos_pan[idx], ann_info)

    def _parse_ann_info_pan(self, img_info, ann_info):
        """Parse annotations and load panoptic ground truths.

        Args:
            img_info (int): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_mask_infos = []
        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label_pan[category_id]

            is_thing = self.coco_pan.loadCats(ids=category_id)[0]['isthing']
            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                if not is_crowd:
                    gt_bboxes.append(bbox)
                    gt_labels.append(contiguous_cat_id)
                else:
                    gt_bboxes_ignore.append(bbox)
                    is_thing = False

            mask_info = {
                'id': ann['id'],
                'category': contiguous_cat_id,
                'is_thing': is_thing
            }
            gt_mask_infos.append(mask_info)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
            seg_map=img_info['segm_file'])
        return ann

    def _filter_imgs_pan(self, min_size=32):
        """Filter images too small or without ground truths."""
        ids_with_ann = []
        # check whether images have legal thing annotations.
        for lists in self.coco_pan.anns.values():
            for item in lists:
                category_id = item['category_id']
                is_thing = self.coco_pan.loadCats(
                    ids=category_id)[0]['isthing']
                if not is_thing:
                    continue
                ids_with_ann.append(item['image_id'])
        ids_with_ann = set(ids_with_ann)

        valid_inds = []
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos_pan):
            img_id = self.img_ids_pan[i]
            if self.filter_empty_gt and img_id not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids_pan = valid_img_ids
        return valid_inds

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos_pan[idx]
        ann_info = self.get_ann_info_pan(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _set_group_flag_pan(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos_pan[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _pan2json(self, results):
        """Convert panoptic results to COCO panoptic json style."""
        label2cat = dict((v, k) for (k, v) in self.cat2label_pan.items())

        pred_annotations = []
        outdir = os.path.join(os.path.dirname(self.outfile_prefix), 'panoptic')
        for idx in range(len(self)):
            img_id = self.img_ids_pan[idx]
            segm_file = self.data_infos_pan[idx]['segm_file']
            pan = results[idx]

            pan_labels = np.unique(pan)
            segm_info = []
            for pan_label in pan_labels:
                sem_label = pan_label % INSTANCE_OFFSET
                # We reserve the length of self.CLASSES for VOID label
                if sem_label == len(self.thing_classes + self.stuff_classes):
                    continue
                # convert sem_label to json label
                cat_id = label2cat[sem_label]
                is_thing = self.categories_pan[cat_id]['isthing']
                mask = pan == pan_label
                area = mask.sum()
                segm_info.append({
                    'id': int(pan_label),
                    'category_id': cat_id,
                    'isthing': is_thing,
                    'area': int(area)
                })
            # evaluation script uses 0 for VOID label.
            pan[pan % INSTANCE_OFFSET == len(self.thing_classes +
                                             self.stuff_classes)] = VOID
            pan = id2rgb(pan).astype(np.uint8)
            mmcv.imwrite(pan[:, :, ::-1], os.path.join(outdir, segm_file))
            record = {
                'image_id': img_id,
                'segments_info': segm_info,
                'file_name': segm_file
            }
            pred_annotations.append(record)
        pan_json_results = dict(annotations=pred_annotations)
        return pan_json_results

    def results2json(self, results):
        """Dump the results to a COCO style json file.

        There are 4 types of results: proposals, bbox predictions, mask
        predictions, panoptic segmentation predictions, and they have
        different data types. This method will automatically recognize
        the type, and dump them to json files.

        .. code-block:: none

            [
                {
                    'pan_results': np.array, # shape (h, w)
                    # ins_results which includes bboxes and RLE encoded masks
                    # is optional.
                    'ins_results': (list[np.array], list[list[str]])
                },
                ...
            ]

        Args:
            results (list[dict]): Testing results of the dataset.

        Returns:
            dict[str: str]: Possible keys are "panoptic", "bbox", "segm", \
                "proposal", and values are corresponding filenames.
        """
        result_files = dict()
        # panoptic segmentation results

        if 'pan_results' in results:
            pan_results = results['pan_results']
            pan_json_results = self._pan2json(pan_results)
            result_files['panoptic'] = f'{self.outfile_prefix}.panoptic.json'
            mmcv.dump(pan_json_results, result_files['panoptic'])

        return result_files

    def get_gt_json(self, result_files):
        """get input for coco panptic evaluation

        Args:
            result_files (dict): path of predict result

        Returns:
            gt_json (dict): gt label
            gt_folder (str): path of gt file
            pred_json(dict): predict result
            pred_folder(str): path of pred file
            categories(dict): panoptic categories
        """

        imgs = self.coco_pan.imgs
        gt_json = self.coco_pan.imgToAnns
        gt_json = [{
            'image_id': k,
            'segments_info': v,
            'file_name': imgs[k]['segm_file']
        } for k, v in gt_json.items()]
        pred_json = mmcv.load(result_files['panoptic'])
        pred_json = dict(
            (el['image_id'], el) for el in pred_json['annotations'])

        gt_folder = self.seg_prefix
        pred_folder = os.path.join(
            os.path.dirname(self.outfile_prefix), 'panoptic')
        categories = self.categories_pan
        return gt_json, gt_folder, pred_json, pred_folder, categories
