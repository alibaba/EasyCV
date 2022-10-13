import logging
import os
import pathlib
import re

import oss2
from hpo_tools.core.utils.config_utils import parse_config


def get_bucket(ori_filepath, oss_config=None):
    # oss_config is the dict or the filepath,such as:
    # {'accessKeyID':'xxx','accessKeySecret':'xxx','endpoint':'xxx'}
    if oss_config is None:
        oss_config = parse_config(
            os.path.join(os.environ['HOME'], '.ossutilconfig'))
    elif isinstance(oss_config, str) and os.path.exists(oss_config):
        oss_config = parse_config(oss_config)
    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    cname = oss_config['endpoint']
    oss_pattern = re.compile(r'oss://([^/]+)/(.+)')
    m = oss_pattern.match(ori_filepath)
    if not m:
        raise IOError('invalid oss path: ' + ori_filepath +
                      ' should be oss://<bucket_name>/path')
    bucket_name, path = m.groups()
    path = path.replace('//', '/')
    bucket_name = bucket_name.split('.')[0]
    logging.info('bucket_name: %s, path: %s', bucket_name, path)

    bucket = oss2.Bucket(auth, cname, bucket_name)

    return bucket, path


def copy_oss_dir(bucket, path, dst_filepath):
    for b in oss2.ObjectIteratorV2(bucket, path, delimiter='/'):
        print(b.key)
        if not b.is_prefix():
            file_name = b.key[b.key.rindex('/') + 1:]
            if len(file_name):
                filepath = os.path.join(dst_filepath,
                                        b.key[:b.key.rindex('/')])
                pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
                logging.info('downloadfile--> %s', b.key)
                bucket.get_object_to_file(b.key,
                                          os.path.join(dst_filepath, b.key))
        else:
            copy_oss_dir(bucket, b.key, dst_filepath)


def copy_dir(ori_filepath, dst_filepath, oss_config=None):
    logging.info('start copy from %s to %s', ori_filepath, dst_filepath)
    bucket, path = get_bucket(ori_filepath=ori_filepath, oss_config=oss_config)
    copy_oss_dir(bucket=bucket, path=path, dst_filepath=dst_filepath)
    logging.info('copy end')


def copy_file(ori_filepath, dst_filepath, oss_config=None):
    logging.info('start copy from %s to %s', ori_filepath, dst_filepath)
    bucket, path = get_bucket(ori_filepath=ori_filepath, oss_config=oss_config)
    bucket.get_object_to_file(path, dst_filepath)


def upload_file(ori_filepath, dst_filepath, oss_config=None):
    logging.info('start upload to %s from %s', ori_filepath, dst_filepath)
    bucket, path = get_bucket(ori_filepath=ori_filepath, oss_config=oss_config)
    bucket.put_object_from_file(path, dst_filepath)
