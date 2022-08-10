# Copyright (c) Alibaba, Inc. and its affiliates.
import configparser
import logging
import os
import sys
import urllib
from collections import namedtuple
from contextlib import contextmanager
from io import StringIO

from tqdm import tqdm

OSS_PREFIX = 'oss://'
URL_PREFIX = 'https://'


def create_namedtuple(**kwargs):
    return namedtuple('namedtuple', list(kwargs.keys()))(**kwargs)


def is_oss_path(s):
    return s.startswith(OSS_PREFIX)


def is_url_path(s):
    return s.startswith(URL_PREFIX)


def url_path_exists(url):
    try:
        urllib.request.urlopen(url).code
    except Exception as err:
        print(err)
    return True


def get_oss_config():
    """
    Get oss config file from env `OSS_CONFIG_FILE`,
    default file is `~/.ossutilconfig`.
    """
    oss_cfg_file = os.environ.get('OSS_CONFIG_FILE', '~/.ossutilconfig')
    oss_cfg_file = os.path.expanduser(oss_cfg_file)

    if not os.path.isabs(oss_cfg_file):
        raise ValueError('Not support relative path for `OSS_CONFIG_FILE`!')

    if not os.path.exists(oss_cfg_file):
        raise ValueError(
            'Please add the oss config file and add `[Bucket-Endpoint]` in your oss config file, '
            'refer to: https://help.aliyun.com/document_detail/120072.html')

    cfg_parser = configparser.ConfigParser()
    cfg_parser.read(oss_cfg_file)

    if 'Bucket-Endpoint' not in cfg_parser:
        raise ValueError(
            'Please add `[Bucket-Endpoint]` in your oss config file, refer to: https://help.aliyun.com/document_detail/120072.html'
        )

    credential = dict(cfg_parser['Credentials'])
    bucket_endpoint = dict(cfg_parser['Bucket-Endpoint'])
    buckets = list(bucket_endpoint.keys())
    hosts = list(bucket_endpoint.values())

    oss_config = dict(
        ak_id=credential['accesskeyid'],
        ak_secret=credential['accesskeysecret'],
        hosts=hosts,
        buckets=buckets)

    return oss_config


@contextmanager
def oss_progress(desc):
    progress = None

    def callback(i, n):
        nonlocal progress
        if progress is None:
            progress = tqdm(
                total=n,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
                desc=desc,
                mininterval=1.0,
                maxinterval=5.0)
        progress.update(i - progress.n)

    yield callback
    if progress is not None:
        progress.close()


@contextmanager
def ignore_oss_error(msg=''):
    import oss2
    try:
        yield
    except (oss2.exceptions.RequestError, oss2.exceptions.ServerError) as e:
        logging.error(str(e) + ' ' + msg)


@contextmanager
def mute_stderr():
    cache = sys.stderr
    sys.stderr = StringIO()
    try:
        yield None
    finally:
        sys.stderr = cache
