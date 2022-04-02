# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bninceptionPre(image, mean=[104, 117, 128], std=[1, 1, 1]):
    """
    Args:
        image: pytorch Image tensor from PIL (range 0~1), bgr format
        mean : norm mean
        std  : norm val
    Returns:
        A image norm in 0~255, rgb format
    """
    expand_batch_dim = len(image.size()) == 3
    if expand_batch_dim:
        image = image.unsqueeze(0)
    image = image * 255
    image = image[:, [2, 1, 0]]
    for i in range(2):
        image[:, i, ...] -= mean[i]
        image[:, i, ...] /= std[i]
    return image


def randomErasing(image,
                  probability=0.5,
                  sl=0.02,
                  sh=0.2,
                  r1=0.3,
                  mean=[0.4914, 0.4822, 0.4465]):
    expand_batch_dim = len(image.size()) == 3
    if expand_batch_dim:
        image = image.unsqueeze(0)

    batch_size = image.size(0)
    width = image.size(2)
    height = image.size(3)
    area = width * height

    for index in range(batch_size):
        erase_flag = False

        while not erase_flag:

            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1 / r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = random.randint(0, height - h)
                y1 = random.randint(0, width - w)
                image[index, 0, x1:x1 + h, y1:y1 + w] = mean[0]
                image[index, 1, x1:x1 + h, y1:y1 + w] = mean[1]
                image[index, 2, x1:x1 + h, y1:y1 + w] = mean[2]

                erase_flag = True

    return image


def solarize(tensor, threshold=0.5, apply_prob=0.2):
    """
        tensor : pytorch tensor
    """
    inverted_tensor = threshold * 2 - tensor
    mask = tensor >= threshold

    t = torch.ones_like(mask).type(torch.bool)  # a true mask
    return mask * inverted_tensor + (t ^ mask) * tensor


def gaussianBlurDynamic(image, apply_prob=0.5):
    expand_batch_dim = len(image.size()) == 3
    if expand_batch_dim:
        image = image.unsqueeze(0)

    batch_size = image.size(0)
    result = torch.zeros_like(image, device=image.device)

    for index in range(batch_size):
        if random.random() < apply_prob:
            sigma = random.uniform(0.1, 2.0)
            kernel_size = int(sigma * 4 + 0.5)
            radius = int(kernel_size / 2)
            kernel_size = radius * 2 + 1

            # x = torch.arange(-radius, radius + 1).cuda()
            x = torch.arange(-radius, radius + 1, device=image.device)
            x = x.to(image.dtype)
            blur_filter = torch.exp(-torch.pow(x, 2.0) / (2.0 * (sigma**2)))
            blur_filter = blur_filter.div(blur_filter.sum())
            blur_v = torch.reshape(blur_filter, [1, 1, kernel_size, 1])
            blur_h = torch.reshape(blur_filter, [1, 1, 1, kernel_size])

            num_channels, _, _ = image.size(1), image.size(2), image.size(3)
            blur_h = blur_h.repeat(num_channels, 1, 1, 1)
            blur_v = blur_v.repeat(num_channels, 1, 1, 1)
            pad_length = int((kernel_size - 1) / 2)
            blurred = F.conv2d(
                image[index:index + 1],
                blur_h,
                stride=1,
                padding=(0, pad_length),
                groups=3)
            blurred = F.conv2d(
                blurred, blur_v, stride=1, padding=(pad_length, 0), groups=3)
            if expand_batch_dim:
                blurred = blurred.squeeze(0)
            result[index] = blurred
        else:
            result[index] = image[index]

    return result


def gaussianBlur(image, kernel_size=22, apply_prob=0.5):
    expand_batch_dim = len(image.size()) == 3
    if expand_batch_dim:
        image = image.unsqueeze(0)

    batch_size = image.size(0)
    result = torch.zeros_like(image, device=image.device)

    for index in range(batch_size):
        if random.random() < apply_prob:
            sigma = random.uniform(0.1, 2.0)
            radius = int(kernel_size / 2)
            kernel_size = radius * 2 + 1
            # x = torch.arange(-radius, radius + 1).cuda()
            x = torch.arange(-radius, radius + 1, device=image.device)
            x = x.to(image.dtype)
            blur_filter = torch.exp(-torch.pow(x, 2.0) / (2.0 * (sigma**2)))
            blur_filter = blur_filter.div(blur_filter.sum())
            blur_v = torch.reshape(blur_filter, [1, 1, kernel_size, 1])
            blur_h = torch.reshape(blur_filter, [1, 1, 1, kernel_size])

            num_channels, _, _ = image.size(1), image.size(2), image.size(3)
            blur_h = blur_h.repeat(num_channels, 1, 1, 1)
            blur_v = blur_v.repeat(num_channels, 1, 1, 1)
            pad_length = int((kernel_size - 1) / 2)
            blurred = F.conv2d(
                image[index:index + 1],
                blur_h,
                stride=1,
                padding=(0, pad_length),
                groups=3)
            blurred = F.conv2d(
                blurred, blur_v, stride=1, padding=(pad_length, 0), groups=3)
            if expand_batch_dim:
                blurred = blurred.squeeze(0)
            result[index] = blurred
        else:
            result[index] = image[index]

    return result


def randomGrayScale(image, apply_prob=0.2):
    expand_batch_dim = len(image.size()) == 3
    if expand_batch_dim:
        image = image.unsqueeze(0)
    batch_size = image.size(0)
    for index in range(batch_size):
        if random.random() < apply_prob:
            tmp = 0.299 * image[index, 0] + 0.587 * image[
                index, 1] + 0.114 * image[index, 2]
            image[index, 0] = tmp
            image[index, 1] = tmp
            image[index, 2] = tmp
    return image


def mixUp(image, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = image.size()[0]

    assert batch_size > 2
    assert batch_size % 2 == 0

    mixed_x = lam * image[0:int(batch_size / 2),
                          ...] + (1 - lam) * image[int(batch_size / 2):, ...]

    return mixed_x, lam


def mixUpCls(data, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size(0)
    index = torch.randperm(batch_size, device=data.device)
    data_mixed = lam * data + (1 - lam) * data[index, :]

    return data_mixed, lam, index


if __name__ == '__main__':
    a = torch.ones([12, 3, 224, 224])
    print(torch.sum(a))
    f = eval('gaussianBlur')
    print(type(f))
    b = f(a)
    print(torch.sum(b))

    print(b.shape)
