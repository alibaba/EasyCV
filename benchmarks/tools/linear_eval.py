# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import shutil
import time

import torch

from easycv.framework.errors import ValueError

args = argparse.ArgumentParser(description='Process some integers.')
args.add_argument(
    'model_path',
    type=str,
    help='linear eval model path',
    nargs='?',
    default='')
args.add_argument(
    'work_dirname',
    type=str,
    help='evaluation work dir name',
    nargs='?',
    default='tmp_evaluation')
args.add_argument(
    'work_dirroot',
    type=str,
    help='evaluation work dir root',
    nargs='?',
    default='work_dirs/benchmarks/linear_classification/imagenet')
args.add_argument(
    'eval_config',
    type=str,
    help='evaluation work dir name',
    nargs='?',
    default='configs/benchmarks/linear_classification/imagenet/tmp_feature.py')

TIME_LOG = []


def timelog(func):
    if type(func) == str:
        time_log = 'time_log %s : %s' % (time.asctime(
            time.localtime(time.time())), func)
        TIME_LOG.append(time_log)
        return

    def wrapper(*args, **kwargs):
        time_log = 'time_log %s : %s' % (time.asctime(
            time.localtime(time.time())), func.__name__)
        TIME_LOG.append(time_log)
        print(time_log)
        return func(*args, **kwargs)

    return wrapper


@timelog
def move_file(model_path, work_dir):
    if model_path[-4:] != '.pth':
        print('model path is Invalid!')
        exit()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    filename = model_path.split('/')[-1]
    filename = os.path.join(work_dir, filename)

    if os.path.exists(filename):
        return filename

    if model_path[:3] == 'oss':
        os.system('ossutil64 cp -f %s %s' % (model_path, work_dir))
    else:
        shutil.copy(model_path, work_dir)

    return filename


@timelog
def extract_model(model_path):
    backbone_file = os.path.join(*(['/'] + model_path.split('/')[:-1] +
                                   ['backbone.pth']))

    ck = torch.load(model_path, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict())
    has_backbone = False
    for key, value in ck['state_dict'].items():
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
            has_backbone = True
    if not has_backbone:
        raise ValueError('Cannot find a backbone module in the checkpoint.')
    torch.save(output_dict, backbone_file)

    return backbone_file


@timelog
def extract_feature(project_path, work_dir, backbone_file):
    os.chdir(project_path)
    os.system(
        "PORT=29513 bash tools/dist_extract.sh configs/classification/imagenet/r50_extract.py 8  \
    %s \
    --pretrained=%s \
    --layer-ind=\'4\' --dataset-config benchmarks/extract_info/imagenet.py" %
        (work_dir, backbone_file))

    return


def modify_config_file(config_file, keywords):
    lines = open(config_file).readlines()
    data = ''
    for l in lines:
        for k in keywords.keys():
            # match keywords
            if ('%s=' % k in l or '%s =' % k in l) and '#' not in l:
                # in order to match the space before k= in l
                idx = max(l.find('%s =' % k), l.find('%s=' % k))
                l = l[:idx] + "%s=\'%s\'\n" % (k, keywords[k])
        data += l

    f = open(config_file, 'w')
    f.write(data)
    f.close()
    return


@timelog
def linear_eval(project_path, feature_path, config_file):
    os.chdir(project_path)
    keywords = {
        'data_root_path': feature_path,
    }
    modify_config_file(config_file, keywords)
    os.system('sh tools/dist_train.sh %s 8' % config_file)
    return


if __name__ == '__main__':
    project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    args = args.parse_args()
    model_path = args.model_path
    work_dirname = args.work_dirname
    work_dirroot = os.path.join(project_path, args.work_dirroot)

    work_dir = os.path.join(work_dirroot, work_dirname)
    print('model_path   : %s' % model_path)
    print('work_dirname : %s' % work_dirname)
    print('work_dir     : %s' % work_dir)

    model_path = move_file(model_path, work_dir)

    backbone_file = extract_model(model_path)

    extract_feature(project_path, work_dir, backbone_file)

    linear_eval(project_path, os.path.join(work_dir, 'features'),
                args.eval_config)

    timelog('end')

    for l in TIME_LOG:
        print(l)
