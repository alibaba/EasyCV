# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
from multiprocessing import cpu_count

import numpy as np

from easycv.datasets.detection.data_sources.base import DetSourceBase
from easycv.datasets.registry import DATASOURCES
from easycv.file import io


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


def parser_manifest_row_str(row_str, classes):
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

    bboxes, gt_labels = [], []
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
                gt_labels.append(classes.index(class_name[0]))
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
                gt_labels.append(classes.index(class_name))
        break

    parse_results['filename'] = img_url
    parse_results['gt_bboxes'] = np.array(bboxes, dtype=np.float32)
    parse_results['gt_labels'] = np.array(gt_labels, dtype=np.int64)

    return parse_results


@DATASOURCES.register_module
class DetSourcePAI(DetSourceBase):
    """
    data format please refer to: https://help.aliyun.com/document_detail/311173.html
    """

    def __init__(self,
                 path,
                 classes=[],
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 parse_fn=parser_manifest_row_str,
                 num_processes=int(cpu_count() / 2),
                 **kwargs):
        """
        Args:
            path: Path of manifest path with pai label format
            classes: classes list
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            parse_fn: parse function to parse item of source iterator
            num_processes: number of processes to parse samples
        """

        self.manifest_path = path
        super(DetSourcePAI, self).__init__(
            classes=classes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly,
            parse_fn=parse_fn,
            num_processes=num_processes)

    def get_source_iterator(self):
        with io.open(self.manifest_path, 'r') as f:
            rows = f.read().splitlines()
        return rows
