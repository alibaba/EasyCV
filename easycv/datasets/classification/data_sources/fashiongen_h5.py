# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import h5py
from PIL import Image

from easycv.datasets.registry import DATASOURCES
from easycv.datasets.utils import tfrecord_util
from easycv.file import io
from easycv.utils.dist_utils import dist_zero_exec

H5_KEYS = ['input_category', 'input_image', 'input_subcategory']
H5_LABEL_LIST = [
    b'SHIRTS', b'SWEATERS', b'JEANS', b'PANTS', b'TOPS', b'SUITS & BLAZERS',
    b'SHORTS', b'JACKETS & COATS', b'TIES', b'HATS', b'SKIRTS', b'JUMPSUITS',
    b'SWIMWEAR', b'DRESSES', b'BELTS & SUSPENDERS', b'LINGERIE', b'SCARVES',
    b'GLOVES', b'FINE JEWELRY', b'CLUTCHES & POUCHES', b'BLANKETS', b'JEWELRY',
    b'BACKPACKS', b'SHOULDER BAGS', b'UNDERWEAR & LOUNGEWEAR', b'KEYCHAINS',
    b'TOTE BAGS', b'BOAT SHOES & MOCCASINS', b'POUCHES & DOCUMENT HOLDERS',
    b'SNEAKERS', b'DUFFLE & TOP HANDLE BAGS', b'EYEWEAR', b'BOOTS', b'FLATS',
    b'LACE UPS', b'MONKSTRAPS', b'LOAFERS', b'SOCKS',
    b'POCKET SQUARES & TIE BARS', b'SANDALS', b'HEELS',
    b'MESSENGER BAGS & SATCHELS', b'ESPADRILLES', b'DUFFLE BAGS',
    b'BAG ACCESSORIES', b'TRAVEL BAGS', b'MESSENGER BAGS', b'BRIEFCASES'
]


@DATASOURCES.register_module
class FashionGenH5(object):

    def __init__(self,
                 h5file_path,
                 return_label=True,
                 cache_path='data/fashionGenH5'):

        self.h5file = h5file_path
        self.return_label = return_label
        if tfrecord_util.is_oss_path(self.h5file):
            with dist_zero_exec():
                dst_path = os.path.join(cache_path, h5file_path)
                io.copy(self.h5file, dst_path)
                self.h5file = dst_path

        self.label_list = H5_LABEL_LIST

    def get_length(self):
        return h5py.File(self.h5file, 'r')[H5_KEYS[0]][:].shape[0]

    def get_sample(self, idx):
        with h5py.File(self.h5file, 'r') as db:
            img = db[H5_KEYS[1]][idx]
            name = db[H5_KEYS[0]][idx]
        img = img[..., [2, 1, 0]]  # transfer to RGB
        img = Image.fromarray(img)

        result_dict = {'img': img}

        if self.return_label:
            label = self.label_list.index(name)
            result_dict.update({'gt_labels': label})

        return result_dict
