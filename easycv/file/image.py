# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import time

import cv2
import numpy as np
from PIL import Image

from easycv.file import io
from easycv.utils.constant import MAX_READ_IMAGE_TRY_TIMES
from .utils import is_oss_path


def load_image(img_path, mode='BGR', max_try_times=MAX_READ_IMAGE_TRY_TIMES):
    """Return np.ndarray[unit8]
    """
    # TODO: functions of multi tries should be in the `io.open`
    try_cnt = 0
    img = None
    while try_cnt < max_try_times:
        try:
            with io.open(img_path, 'rb') as infile:
                # cv2.imdecode may corrupt when the img is broken
                image = Image.open(infile)  # RGB
                img = np.asarray(image, dtype=np.uint8)
                if mode.upper() == 'BGR':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                assert mode.upper() in ['RGB', 'BGR'
                                        ], 'Only support `RGB` and `BGR` mode!'
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
        raise ValueError('Read Image Error: ' + img_path)

    return img
