# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import wget

# The location where downloaded data is stored
from easycv.utils.constant import CACHE_DIR


def download(link, target_dir=CACHE_DIR):
    file_name = wget.filename_from_url(link)
    # 查看是否有压缩包，无压缩包的话下载压缩包
    if not os.path.exists(os.path.join(target_dir, file_name)):
        try:
            print(f'{file_name} is start downlaod........')
            file_name = wget.download(link, out=target_dir)
            print(f'{file_name} is download finished\n')
        except:
            print(f'{file_name} is download fail')
            exit()
    # The prevention of Ctrol + C
    if not os.path.exists(os.path.join(target_dir, file_name)):
        exit()
    return os.path.join(target_dir, file_name)


def check_path_exists(map_path):
    for value in map_path.values():
        assert os.path.exists(value), f'{value} is not exists'
    return map_path
