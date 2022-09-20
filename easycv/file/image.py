# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import logging
import time

import cv2
import numpy as np
from PIL import Image

from easycv import file
from easycv.framework.errors import IOError
from easycv.utils.constant import MAX_READ_IMAGE_TRY_TIMES
from .utils import is_oss_path, is_url_path


def load_image(img_path, mode='BGR', max_try_times=MAX_READ_IMAGE_TRY_TIMES):
    """Return np.ndarray[unit8]
    """
    # TODO: functions of multi tries should be in the `io.open`
    try_cnt = 0
    img = None
    while try_cnt < max_try_times:
        try:
            if is_url_path(img_path):
                from mmcv.fileio.file_client import HTTPBackend
                client = HTTPBackend()
                img_bytes = client.get(img_path)
                buff = io.BytesIO(img_bytes)
                image = Image.open(buff)
                if mode.upper() != 'BGR' and image.mode.upper() != mode.upper(
                ):
                    image = image.convert(mode.upper())
                img = np.asarray(image, dtype=np.uint8)
            else:
                with file.io.open(img_path, 'rb') as infile:
                    # cv2.imdecode may corrupt when the img is broken
                    image = Image.open(infile)
                    if mode.upper() != 'BGR' and image.mode.upper(
                    ) != mode.upper():
                        image = image.convert(mode.upper())
                    img = np.asarray(image, dtype=np.uint8)

            if mode.upper() == 'BGR':
                if image.mode.upper() != 'RGB':
                    image = image.convert('RGB')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            assert img is not None
            break
        except Exception as e:
            logging.error(e)
            logging.warning('Read file {} fault, try count : {}'.format(
                img_path, try_cnt))
            # frequent access to oss will cause error, sleep can aviod it
            if is_oss_path(img_path):
                sleep_time = 1
                logging.warning(
                    'Sleep {}s, frequent access to oss file may cause error.'.
                    format(sleep_time))
                time.sleep(sleep_time)
        try_cnt += 1

    if img is None:
        raise IOError('Read Image Error: ' + img_path)

    return img
