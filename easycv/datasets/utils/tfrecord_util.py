# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os

from easycv.file import io
from easycv.utils import dist_utils


def get_imagenet_dali_tfrecord_feature():
    import nvidia.dali.tfrecord as tfrec

    imagenet_feature = {
        'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ''),
        'image/format': tfrec.FixedLenFeature((), tfrec.string, 'jpeg'),
        'image/class/label': tfrec.FixedLenFeature([], tfrec.int64, -1),
    }
    return imagenet_feature


def get_path_and_index(file_list_or_path):
    if type(file_list_or_path) == str:
        lines = io.open(file_list_or_path).readlines()
    else:
        lines = file_list_or_path
    path = []
    index = []
    for i in lines:
        i = i.strip()
        if i.endswith('.idx') or i.endswith('.info'):
            pass
        else:
            path.append(i)
            index.append(i + '.idx')
    return path, index


# multi worker download using oss
def download_tfrecord(file_list_or_path,
                      target_path,
                      slice_count=1,
                      slice_id=0,
                      force=False):
    """Download data from oss.
    Use the processes on the gpus to slice download, each gpu process downloads part of the data.
    The number of slices is the same as the number of gpu processes.
    Support tfrecord of ImageNet style.
    tfrecord_dir
        |---train1
        |---train1.idx
        |---train2
        |---train2.idx
        |---...

    Args:
        file_list_or_path:  A list of absolute data path or a path str
                    type(file_list) == list means this is the list
                    type(file_list) == str means open(file_list).readlines()
        target_path: A str, download path
        slice_count: Download worker num
        slice_id : Download worker ID
        force: If false, skip download if the file already exists in the target path.
            If true, recopy and replace the original file.

    Returns:
        path: list of str,  download tfrecord path
        index_path: list of str, download tfrecord idx path
    """
    with dist_utils.dist_zero_exec():
        if not os.path.exists(target_path):
            os.makedirs(target_path)

    logging.info(f'num gpu(slice_count): {slice_count}')

    if isinstance(file_list_or_path, list):
        all_file_list = file_list_or_path
    else:
        with io.open(file_list_or_path, 'r') as f:
            lines = f.readlines()
        all_file_list = [i.strip() for i in lines]

    all_data_list = [
        all_file_list[i] for i in range(len(all_file_list))
        if not all_file_list[i].endswith('.idx')
        and not all_file_list[i].endswith('.info')
    ]
    all_index_list = [
        all_file_list[i] for i in range(len(all_file_list))
        if all_file_list[i].endswith('.idx')
    ]
    if not all_index_list:
        all_index_list = [i + '.idx' for i in all_data_list]

    idx = 0
    for data_path in all_data_list:
        # split data list to target worker
        if idx % slice_count == slice_id:
            target_file = os.path.join(target_path,
                                       os.path.split(data_path)[-1])
            if not force and io.exists(target_file):
                logging.info('%s already exists, skip download!' % target_file)
                continue
            io.copy(data_path, target_file)
            logging.info('Finished download file: %s' % data_path)
        idx += 1

    idx = 0
    for idx_path in all_index_list:
        # split data list to target worker
        if idx % slice_count == slice_id:
            target_file = os.path.join(target_path,
                                       os.path.split(idx_path)[-1])
            if not force and io.exists(target_file):
                logging.info('%s already exists, skip download!' % target_file)
                continue
            io.copy(idx_path, target_file)
            logging.info('Finished download file: %s' % idx_path)
        idx += 1

    logging.info('rank %s finish downloads!' % slice_id)

    dist_utils.barrier()

    # return all data list
    new_path = []
    for data_path in all_data_list:
        target_file = os.path.join(target_path, os.path.split(data_path)[-1])
        new_path.append(target_file)
    all_data_list = new_path

    new_index_path = []
    for idx_path in all_index_list:
        target_file = os.path.join(target_path, os.path.split(idx_path)[-1])
        new_index_path.append(target_file)
    all_index_list = new_index_path

    return all_data_list, all_index_list
