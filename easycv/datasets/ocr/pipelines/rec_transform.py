# Modified from https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/ppocr/data/imaug
import math
import random
import sys

import cv2
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from easycv.datasets.registry import PIPELINES


@PIPELINES.register_module()
class RecConAug(object):
    """concat multiple texts together for text recognition training
    """

    def __init__(self,
                 prob=0.5,
                 image_shape=(32, 320, 3),
                 max_text_length=25,
                 **kwargs):
        """

        Args:
            prob (float, optional): the probability whether do data augmentation. Defaults to 0.5.
            image_shape (tuple, optional): the output image shape. Defaults to (32, 320, 3).
            max_text_length (int, optional): the max length of text label. Defaults to 25.
        """
        self.prob = prob
        self.max_text_length = max_text_length
        self.image_shape = image_shape
        self.max_wh_ratio = self.image_shape[1] / self.image_shape[0]

    def merge_ext_data(self, data, ext_data):
        ori_w = round(data['img'].shape[1] / data['img'].shape[0] *
                      self.image_shape[0])
        ext_w = round(ext_data['img'].shape[1] / ext_data['img'].shape[0] *
                      self.image_shape[0])
        data['img'] = cv2.resize(data['img'], (ori_w, self.image_shape[0]))
        ext_data['img'] = cv2.resize(ext_data['img'],
                                     (ext_w, self.image_shape[0]))
        data['img'] = np.concatenate([data['img'], ext_data['img']], axis=1)
        data['label'] += ext_data['label']
        return data

    def __call__(self, data):
        rnd_num = random.random()
        if rnd_num > self.prob:
            return data
        for idx, ext_data in enumerate(data['ext_data']):
            if len(data['label']) + len(
                    ext_data['label']) > self.max_text_length:
                break
            concat_ratio = data['img'].shape[1] / data['img'].shape[
                0] + ext_data['img'].shape[1] / ext_data['img'].shape[0]
            if concat_ratio > self.max_wh_ratio:
                break
            data = self.merge_ext_data(data, ext_data)
        data.pop('ext_data')
        return data


@PIPELINES.register_module()
class RecAug(object):
    """data augmentation function for ocr recognition
    """

    def __init__(self, use_tia=True, aug_prob=0.4, **kwargs):
        """

        Args:
            use_tia (bool, optional): whether make tia augmentation. Defaults to True.
            aug_prob (float, optional): the probability were do data augmentation. Defaults to 0.4.
        """
        self.use_tia = use_tia
        self.aug_prob = aug_prob

    def __call__(self, data):
        img = data['img']
        img = warp(img, 10, self.use_tia, self.aug_prob)
        data['img'] = img
        return data


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def cvtColor(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """

    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


class Config:
    """
    Config
    """

    def __init__(self, use_tia):
        self.anglex = random.random() * 30
        self.angley = random.random() * 15
        self.anglez = random.random() * 10
        self.fov = 42
        self.r = 0
        self.shearx = random.random() * 0.3
        self.sheary = random.random() * 0.05
        self.borderMode = cv2.BORDER_REPLICATE
        self.use_tia = use_tia

    def make(self, w, h, ang):
        """
        make
        """
        self.anglex = random.random() * 5 * flag()
        self.angley = random.random() * 5 * flag()
        self.anglez = -1 * random.random() * int(ang) * flag()
        self.fov = 42
        self.r = 0
        self.shearx = 0
        self.sheary = 0
        self.borderMode = cv2.BORDER_REPLICATE
        self.w = w
        self.h = h

        self.perspective = self.use_tia
        self.stretch = self.use_tia
        self.distort = self.use_tia

        self.crop = True
        self.affine = False
        self.reverse = True
        self.noise = True
        self.jitter = True
        self.blur = True
        self.color = True


def rad(x):
    """
    rad
    """
    return x * np.pi / 180


def get_warpR(config):
    """
    get_warpR
    """
    anglex, angley, anglez, fov, w, h, r = \
        config.anglex, config.angley, config.anglez, config.fov, config.w, config.h, config.r
    if w > 69 and w < 112:
        anglex = anglex * 1.5

    z = np.sqrt(w**2 + h**2) / 2 / np.tan(rad(fov / 2))
    # Homogeneous coordinate transformation matrix
    rx = np.array(
        [[1, 0, 0, 0], [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
         [0, -np.sin(rad(anglex)),
          np.cos(rad(anglex)), 0], [0, 0, 0, 1]], np.float32)
    ry = np.array([[np.cos(rad(angley)), 0,
                    np.sin(rad(angley)), 0], [0, 1, 0, 0],
                   [
                       -np.sin(rad(angley)),
                       0,
                       np.cos(rad(angley)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    rz = np.array(
        [[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
         [-np.sin(rad(anglez)),
          np.cos(rad(anglez)), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    r = rx.dot(ry).dot(rz)
    # generate 4 points
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = np.array([dst1, dst2, dst3, dst4])
    org = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    dst = np.zeros((4, 2), np.float32)
    # Project onto the image plane
    dst[:, 0] = list_dst[:, 0] * z / (z - list_dst[:, 2]) + pcenter[0]
    dst[:, 1] = list_dst[:, 1] * z / (z - list_dst[:, 2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    dst1, dst2, dst3, dst4 = dst
    r1 = int(min(dst1[1], dst2[1]))
    r2 = int(max(dst3[1], dst4[1]))
    c1 = int(min(dst1[0], dst3[0]))
    c2 = int(max(dst2[0], dst4[0]))

    try:
        ratio = min(1.0 * h / (r2 - r1), 1.0 * w / (c2 - c1))

        dx = -c1
        dy = -r1
        T1 = np.float32([[1., 0, dx], [0, 1., dy], [0, 0, 1.0 / ratio]])
        ret = T1.dot(warpR)
    except:
        ratio = 1.0
        T1 = np.float32([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        ret = T1
    return ret, (-r1, -c1), ratio, dst


def get_warpAffine(config):
    """
    get_warpAffine
    """
    anglez = config.anglez
    rz = np.array(
        [[np.cos(rad(anglez)), np.sin(rad(anglez)), 0],
         [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    return rz


def warp(img, ang, use_tia=True, prob=0.4):
    """
    warp
    """
    h, w, _ = img.shape
    config = Config(use_tia=use_tia)
    config.make(w, h, ang)
    new_img = img

    if config.distort:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = tia_distort(new_img, random.randint(3, 6))

    if config.stretch:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = tia_stretch(new_img, random.randint(3, 6))

    if config.perspective:
        if random.random() <= prob:
            new_img = tia_perspective(new_img)

    if config.crop:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = get_crop(new_img)

    if config.blur:
        if random.random() <= prob:
            new_img = blur(new_img)
    if config.color:
        if random.random() <= prob:
            new_img = cvtColor(new_img)
    if config.jitter:
        new_img = jitter(new_img)
    if config.noise:
        if random.random() <= prob:
            new_img = add_gasuss_noise(new_img)
    if config.reverse:
        if random.random() <= prob:
            new_img = 255 - new_img
    return new_img


class WarpMLS:

    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 *
                                                      (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)

        if self.pt_count < 2:
            return

        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break

            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break

                    w[k] = 1. / ((i - self.dst_pts[k][0]) *
                                 (i - self.dst_pts[k][0]) +
                                 (j - self.dst_pts[k][1]) *
                                 (j - self.dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])

                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue

                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = np.sum(
                            pt_i * cur_pt) * self.src_pts[k][0] - np.sum(
                                pt_j * cur_pt) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][
                            0] + np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]

                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdx[i, j], self.rdx[i,
                                                                          nj],
                                                 self.rdx[ni, j], self.rdx[ni,
                                                                           nj])
                delta_y = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdy[i, j], self.rdy[i,
                                                                          nj],
                                                 self.rdy[ni, j], self.rdy[ni,
                                                                           nj])
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h,
                    j:j + w] = self.__bilinear_interp(x, y, self.src[nyi, nxi],
                                                      self.src[nyi, nxi1],
                                                      self.src[nyi1, nxi],
                                                      self.src[nyi1, nxi1])

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst


def tia_distort(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh),
         np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append(
        [np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            np.random.randint(thresh) - half_thresh
        ])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            img_h + np.random.randint(thresh) - half_thresh
        ])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_stretch(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


@PIPELINES.register_module()
class RecResizeImg(object):

    def __init__(
            self,
            image_shape,
            infer_mode=False,
            character_dict_path='./easycv/datasets/ocr/dict/ppocr_keys_v1.txt',
            padding=True,
            **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_dict_path = character_dict_path
        self.padding = padding

    def __call__(self, data):
        img = data['img']
        if self.infer_mode and self.character_dict_path is not None:
            norm_img, valid_ratio = resize_norm_img_chinese(
                img, self.image_shape)
        else:
            norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                    self.padding)
        data['img'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


def resize_norm_img(img, image_shape, padding=True):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        # resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image = resized_image / 255
    resized_image -= 0.5
    resized_image /= 0.5
    # padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    # padding_im[:, :, 0:resized_w] = resized_image
    padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
    padding_im[:, 0:resized_w, :] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(imgH * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        # resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image = resized_image / 255
    resized_image -= 0.5
    resized_image /= 0.5
    # padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    # padding_im[:, :, 0:resized_w] = resized_image
    padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
    padding_im[:, 0:resized_w, :] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


@PIPELINES.register_module()
class ClsResizeImg(object):

    def __init__(self, img_shape, **kwargs):
        self.img_shape = img_shape

    def __call__(self, data):
        img = data['img']
        norm_img, _ = resize_norm_img(img, self.img_shape)
        data['img'] = norm_img
        return data
