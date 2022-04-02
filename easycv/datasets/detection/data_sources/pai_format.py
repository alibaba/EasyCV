# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging

import numpy as np
from mmcv.runner.dist_utils import get_dist_info

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from .voc import DetSourceVOC


def get_prior_task_id(keys):
    """"The task id ends with `check` is the highest priority.
    """
    k_list = []
    check_k_list = []
    verify_k_list = []
    for k in keys:
        if k.startswith('label-'):
            if k.endswith('check'):
                check_k_list.append(k)
            elif k.endswith('verify'):
                verify_k_list.append(k)
            else:
                k_list.append(k)

    if len(check_k_list):
        return check_k_list
    if len(k_list):
        return k_list
    if len(verify_k_list):
        return verify_k_list

    return []


def is_itag_v2(row):
    """
    The keyword of the data source is `picUrl` in v1, but is `source` in v2
    """
    if 'source' in row['data']:
        return True
    return False


def parser_manifest_row_str(row_str):
    row = json.loads(row_str.strip())
    _is_itag_v2 = is_itag_v2(row)

    parse_results = {}

    # check img_url
    img_url = row['data']['source']
    if img_url.startswith(('http://', 'https://')):
        logging.warning(
            'Not support http url, only support `oss://`, skip the sample: %s!'
            % img_url)
        return parse_results

    # check task ids
    if _is_itag_v2:
        task_ids = get_prior_task_id(row.keys())
    else:
        task_ids = [
            row_k for row_k in row.keys() if row_k.startswith('label-')
        ]
    if len(task_ids) > 1:
        raise NotImplementedError('Not support multi label task ids: %s!' %
                                  task_ids)
    if not len(task_ids):
        logging.warning('Not find label task id in sample: %s, skip it!' %
                        img_url)
        return parse_results

    ann_json = row[task_ids[0]]
    if not ann_json:
        return parse_results

    bboxes, class_names = [], []
    for result in ann_json['results']:
        if result['type'] != 'image':
            continue

        objs_list = result['data']

        for obj in objs_list:
            if _is_itag_v2:
                if obj['type'] != 'image/polygon':
                    logging.warning(
                        'Result type should be `image/polygon`, but get %s, skip object %s in %s'
                        % (obj['type'], obj, img_url))
                    continue
                sort_points = sorted(obj['value'], key=sum)
                (x0, y0, x1, y1) = np.concatenate(
                    (sort_points[0], sort_points[-1]), axis=0)
                bboxes.append([x0, y0, x1, y1])
                class_name = list(obj['labels'].values())
                if len(class_name) > 1:
                    raise ValueError(
                        'Not support multi label, get class name  %s!' %
                        class_name)
                class_names.append(class_name[0])
            else:
                if obj['type'] != 'image/rectangleLabel':
                    logging.warning(
                        'result type [%s] in %s is not image/rectangleLabel, skip it!'
                        % (obj['type'], img_url))
                    continue
                value = obj['value']
                x, y, w, h = value['x'], value['y'], value['width'], value[
                    'height']
                bnd = [x, y, x + w, y + h]
                class_name = obj['labels'][0]
                bboxes.append(bnd)
                class_names.append(class_name)
        break

    parse_results['gt_bboxes'] = bboxes
    parse_results['class_names'] = class_names
    parse_results['filename'] = img_url

    return parse_results


@DATASOURCES.register_module
class DetSourcePAI(DetSourceVOC):
    """
    data format please refer to: https://help.aliyun.com/document_detail/311173.html
    """

    def __init__(self,
                 path,
                 classes=[],
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 **kwargs):
        """
        Args:
            path: Path of manifest path with pai label format
            classes: classes list
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
        """
        self.CLASSES = classes
        self.rank, self.world_size = get_dist_info()
        self.manifest_path = path
        self.cache_at_init = cache_at_init
        self.cache_on_the_fly = cache_on_the_fly
        if self.cache_at_init and self.cache_on_the_fly:
            raise ValueError(
                'Only one of `cache_on_the_fly` and `cache_at_init` can be True!'
            )

        with io.open(self.manifest_path, 'r') as f:
            rows = f.read().splitlines()

        self.samples_list = self.build_samples(rows)

    def get_source_info(self, row_str):
        source_info = parser_manifest_row_str(row_str)
        source_info['gt_bboxes'] = np.array(
            source_info['gt_bboxes'], dtype=np.float32)
        source_info['gt_labels'] = np.array([
            self.CLASSES.index(class_name)
            for class_name in source_info['class_names']
        ],
                                            dtype=np.int64)

        return source_info
