# Copyright (c) Alibaba, Inc. and its affiliates.
import io

import cv2
import numpy as np
from cv2 import IMREAD_COLOR
from PIL import Image

from easycv import file
from easycv.framework.errors import KeyError, ValueError
from .utils import is_url_path

try:
    from turbojpeg import TurboJPEG, TJCS_RGB, TJPF_BGR
    turbo_jpeg = TurboJPEG()
    turbo_jpeg_mode = {'RGB': TJCS_RGB, 'BGR': TJPF_BGR}
except:
    turbo_jpeg = None
    turbo_jpeg_mode = None


def load_image_with_pillow(content, mode='BGR', dtype=np.uint8):
    with io.BytesIO(content) as buff:
        image = Image.open(buff)

        if mode.upper() != 'BGR':
            if image.mode.upper() != mode.upper():
                image = image.convert(mode.upper())
            img = np.asarray(image, dtype=dtype)
        else:
            if image.mode.upper() != 'RGB':
                image = image.convert('RGB')
            img = np.asarray(image, dtype=dtype)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def load_image_with_turbojpeg(content, mode='BGR', dtype=np.uint8):
    assert mode.upper() in turbo_jpeg_mode
    if turbo_jpeg is None or turbo_jpeg_mode is None:
        raise ValueError(
            'Please install turbojpeg by "pip install PyTurboJPEG" !')

    img = turbo_jpeg.decode(
        content, pixel_format=turbo_jpeg_mode[mode.upper()])

    if img.dtype != dtype:
        img = img.astype(dtype)

    return img


def load_image_with_cv2(content, mode='BGR', dtype=np.uint8):
    assert mode.upper() in ['BGR', 'RGB']

    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, flags=IMREAD_COLOR)

    if mode.upper() == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != dtype:
        img = img.astype(dtype)

    return img


def _load_image(fp, mode='BGR', dtype=np.uint8, backend='pillow'):
    if backend == 'pillow':
        img = load_image_with_pillow(fp, mode=mode, dtype=dtype)
    elif backend == 'turbojpeg':
        img = load_image_with_turbojpeg(fp, mode=mode, dtype=dtype)
    elif backend == 'cv2':
        img = load_image_with_cv2(fp, mode=mode, dtype=dtype)
    else:
        raise KeyError(
            'Only support backend in ["pillow", "turbojpeg", "cv2"]')
    return img


def load_image(img_path, mode='BGR', dtype=np.uint8, backend='pillow'):
    """Load image file, return np.ndarray.

    Args:
        img_path (str): Image file path.
        mode (str): Order of channel, candidates are `bgr` and `rgb`.
        dtype : Output data type.
        backend (str): The image decoding backend type. Options are `cv2`, `pillow`, `turbojpeg`.
    """
    img = None
    if is_url_path(img_path):
        from mmcv.fileio.file_client import HTTPBackend
        client = HTTPBackend()
        img_bytes = client.get(img_path)
        img = _load_image(img_bytes, mode=mode, dtype=dtype, backend=backend)
    else:
        with file.io.open(img_path, 'rb') as infile:
            img = _load_image(
                infile.read(), mode=mode, dtype=dtype, backend=backend)

    return img
