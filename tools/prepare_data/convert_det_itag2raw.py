# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import functools
import json
import logging
import os
import random
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

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
    image_size = None
    for result in ann_json['results']:
        if result['type'] != 'image':
            continue
        if not image_size:
            image_size = (result['width'], result['height'])
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

    if len(bboxes) == 0:
        bboxes = np.zeros((0, 4), dtype=np.float32)

    parse_results['filename'] = img_url
    parse_results['img_size'] = image_size
    parse_results['gt_bboxes'] = bboxes
    parse_results['gt_labels'] = gt_labels

    return parse_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert itag detection label to raw detection label')
    parser.add_argument(
        'itag_label_file',
        help='itag label manifest file path',
        type=str,
        default=None)
    parser.add_argument(
        'raw_label_dir',
        help='dir path to save raw label',
        type=str,
        default=None)
    parser.add_argument(
        'class_path', help='classname file path', type=str, default=None)
    parser.add_argument(
        '--split',
        action='store_true',
        help='Whether or not split dataset to train/val set')
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=0.3,
        help='The ratio allocated to val set')
    args = parser.parse_args()
    return args


def build_sample(source_item, classes, parse_fn):
    """Build sample info from source item.
    Args:
        source_item: item of source iterator
        classes: classes list
        parse_fn: parse function to parse source_item, only accepts two params: source_item and classes
        load_img: load image or not, if true, cache all images in memory at init
    """
    result_dict = parse_fn(source_item, classes)

    return result_dict


def build_samples(iterable, process_fn):
    samples_list = []
    num_processes = int(cpu_count() / 2)
    with Pool(processes=num_processes) as p:
        with tqdm(total=len(iterable), desc='Scanning images') as pbar:
            for _, result_dict in enumerate(
                    p.imap_unordered(process_fn, iterable)):
                if result_dict:
                    samples_list.append(result_dict)
                pbar.update()

    return samples_list


def get_source_iterator(itag_label_file):
    with io.open(itag_label_file, 'r') as f:
        rows = f.read().splitlines()
    return rows


def parse_class_list(class_path):
    with open(class_path, 'r') as f:
        rows = f.read().splitlines()
        return rows


def write_raw_label(f, sample_dict):
    w, h = sample_dict['img_size'][0], sample_dict['img_size'][1]
    for box, label in zip(sample_dict['gt_bboxes'], sample_dict['gt_labels']):
        plt, prb = box[:2], box[2:]
        cx, cy, bw, bh = (plt[0] + prb[0]) / 2 / w, (
            plt[1] + prb[1]) / 2 / h, (prb[0] - plt[0]) / w, (prb[1] -
                                                              plt[1]) / h
        res = [str(label), str(cx), str(cy), str(bw), str(bh)]
        f.write(' '.join(res) + '\n')


def main():
    args = parse_args()

    class_list = parse_class_list(args.class_path)
    raw_label_dir_train = os.path.join(args.raw_label_dir, 'train')
    raw_label_dir_val = os.path.join(args.raw_label_dir, 'val')
    if not os.path.exists(raw_label_dir_train):
        os.makedirs(raw_label_dir_train)
    if args.split and not os.path.exists(raw_label_dir_val):
        os.makedirs(raw_label_dir_val)

    source_iter = get_source_iterator(args.itag_label_file)
    process_fn = functools.partial(
        build_sample,
        parse_fn=parser_manifest_row_str,
        classes=class_list,
    )
    samples_list = build_samples(source_iter, process_fn)
    random.shuffle(samples_list)
    data_size = len(samples_list)
    for idx, samples_dict in enumerate(samples_list):
        img_path = samples_dict['filename']
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        lable_path = None
        if args.split and idx < args.split_ratio * data_size:
            lable_path = os.path.join(raw_label_dir_val, img_name + '.txt')
        else:
            lable_path = os.path.join(raw_label_dir_train, img_name + '.txt')
        assert not os.path.exists(
            lable_path), 'file %s already exists' % lable_path
        if len(samples_dict['gt_labels']) <= 0:
            continue
        with open(lable_path, 'w') as f:
            write_raw_label(f, samples_dict)


if __name__ == '__main__':
    main()
