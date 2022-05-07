# Copyright (c) OpenMMLab, Alibaba Inc. and its affiliates.
import collections
import copy
import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch

from easycv.datasets.registry import DATASETS, PIPELINES
from easycv.utils.bbox_util import xyxy2xywh as xyxy2cxcywh
from easycv.utils.registry import build_from_cfg
from .raw import DetDataset


@DATASETS.register_module
class DetImagesMixDataset(DetDataset):
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    output boxes format: cx, cy, w, h

    Args:
        data_source (:obj:`DetSourceCoco`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
        label_padding: out labeling padding [N, 120, 5]
    """

    def __init__(self,
                 data_source,
                 pipeline,
                 dynamic_scale=None,
                 skip_type_keys=None,
                 profiling=False,
                 classes=None,
                 yolo_format=True,
                 label_padding=True):

        super(DetImagesMixDataset, self).__init__(
            data_source, pipeline, profiling=profiling, classes=classes)

        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])

        self._skip_type_keys = skip_type_keys

        self.pipeline_yolox = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = build_from_cfg(transform, PIPELINES)
                self.pipeline_yolox.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        if hasattr(self.data_source, 'flag'):
            self.flag = self.data_source.flag

        if dynamic_scale is not None:
            assert isinstance(dynamic_scale, tuple)

        self._dynamic_scale = dynamic_scale

        self.yolo_format = yolo_format
        self.label_padding = label_padding
        self.max_labels_num = 120

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_source.get_sample(idx))
        for (transform, transform_type) in zip(self.pipeline_yolox,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                indexes = transform.get_indexes(self.data_source)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_results = [
                    copy.deepcopy(self.data_source.get_sample(index))
                    for index in indexes
                ]
                results['mix_results'] = mix_results

            if self._dynamic_scale is not None:
                # Used for subsequent pipeline to automatically change
                # the output image size. E.g MixUp, Resize.
                results['scale'] = self._dynamic_scale

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')
            if 'img_scale' in results:
                results.pop('img_scale')

        if self.label_padding:
            cxcywh_gt_bboxes = xyxy2cxcywh(results['gt_bboxes']._data)
            padded_gt_bboxes = torch.zeros((self.max_labels_num, 4),
                                           device=cxcywh_gt_bboxes.device)
            padded_gt_bboxes[range(cxcywh_gt_bboxes.shape[0])[:self.max_labels_num]] = \
                cxcywh_gt_bboxes[:self.max_labels_num].float()

            gt_labels = torch.unsqueeze(results['gt_labels']._data, 1).float()
            padded_labels = torch.zeros((self.max_labels_num, 1),
                                        device=gt_labels.device)
            padded_labels[range(
                gt_labels.shape[0]
            )[:self.max_labels_num]] = gt_labels[:self.max_labels_num]

            results['gt_bboxes'] = padded_gt_bboxes
            results['gt_labels'] = padded_labels

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys

    def update_dynamic_scale(self, dynamic_scale):
        """Update dynamic_scale. It is called by an external hook.

        Args:
            dynamic_scale (tuple[int]): The image scale can be
               changed dynamically.
        """
        assert isinstance(dynamic_scale, tuple)
        self._dynamic_scale = dynamic_scale

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir
