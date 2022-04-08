# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import cv2
import numpy as np
from PIL import Image

from easycv.datasets.registry import DATASOURCES
from easycv.file import io, is_oss_path
from easycv.utils.dist_utils import dist_zero_exec


@DATASOURCES.register_module
class ImageNpy(object):

    def __init__(self, image_file, label_file=None, cache_root='data_cache/'):
        """
            image_file: (local or oss) image data saved in one .npy data  [cv2.img, cv2.img,...]
            label_file: (local or oss) label data saved in one .npy data
        """
        if is_oss_path(image_file):
            with dist_zero_exec():
                dst_path = os.path.join(cache_root, image_file)
                io.copy(image_file, dst_path)
                image_file = dst_path

        self.has_labels = label_file != None
        self.labels = None
        if label_file:
            if is_oss_path(label_file):
                with dist_zero_exec():
                    dst_path = os.path.join(cache_root, label_file)
                    io.copy(label_file, dst_path)
                    label_file = dst_path
            self.labels = np.load(label_file, allow_pickle=True)

        self.data = np.load(image_file, allow_pickle=True)

    def get_length(self):
        return self.data.shape[0]

    def get_sample(self, idx):

        img = Image.fromarray(cv2.cvtColor(self.data[idx], cv2.COLOR_BGR2RGB))

        results = {'img': img}
        if self.labels is not None:
            label = self.labels[idx]
            results.update({'gt_labels': label})

        return results
