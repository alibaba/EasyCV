# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def bound_limits(v):
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return v


def bound_limits_for_list(xywh):
    return tuple([bound_limits(v) for v in xywh])


def xyxy2xywh_with_shape(x, shape):
    dh = 1.0 / shape[0]
    dw = 1.0 / shape[1]
    x_center = (x[0] + x[2]) / 2.0
    y_center = (x[1] + x[3]) / 2.0
    w = x[2] - x[0]  # width
    h = x[3] - x[1]  # height
    x_center *= dw
    y_center *= dh
    w *= dw
    h *= dh

    return bound_limits_for_list((x_center, y_center, w, h))


def batched_cxcywh2xyxy_with_shape(bboxes, shape):
    """reverse of `xyxy2xywh_with_shape`
       transform normalized points `[[x_center, y_center, box_w, box_h],...]`
       to standard [[x1, y1, x2, y2],...]
       Args:
           bboxes: np.array or tensor like [[x_center, y_center, box_w, box_h],...],
               all value is normalized
           shape: img shape: [h, w]
       return: np.array or tensor like [[x1, y1, x2, y2],...]
    """
    h, w = shape[0], shape[1]
    bboxes[:, 0] = bboxes[:, 0] * w  # x_center
    bboxes[:, 1] = bboxes[:, 1] * h  # y_center
    bboxes[:, 2] = bboxes[:, 2] * w  # box w
    bboxes[:, 3] = bboxes[:, 3] * h  # box h

    target = torch.zeros_like(bboxes) if isinstance(bboxes, torch.Tensor) \
        else np.zeros_like(bboxes)
    target[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5  # axis x1
    target[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5  # axis y1
    target[:, 2] = bboxes[:, 0] + bboxes[:, 2] * 0.5  # axis x2
    target[:, 3] = bboxes[:, 1] + bboxes[:, 3] * 0.5  # axis y2

    # handling out-of-bounds
    target[:, 0][target[:, 0] < 0] = 0
    target[:, 0][target[:, 0] > w] = w
    target[:, 1][target[:, 1] < 0] = 0
    target[:, 1][target[:, 1] > h] = h
    target[:, 2][target[:, 2] < 0] = 0
    target[:, 2][target[:, 2] > w] = w
    target[:, 3][target[:, 3] < 0] = 0
    target[:, 3][target[:, 3] > h] = h

    return target


def batched_xyxy2cxcywh_with_shape(bboxes, shape):
    dh = 1.0 / shape[0]
    dw = 1.0 / shape[1]
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 2] * dw
    bboxes[:, 3] = bboxes[:, 3] * dh
    bboxes[:, 0] = bboxes[:, 0] * dw
    bboxes[:, 1] = bboxes[:, 1] * dh
    return bboxes


def xyxy2xywh_coco(bboxes, offset=0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] + offset
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] + offset
    return bboxes


def xywh2xyxy_coco(bboxes, offset=0):
    # Convert nx4 boxes from [x1, y1, w, h] to [x1, y1, y1, y2]
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0] + offset
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1] + offset

    return bboxes


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x_c, y_c, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def bbox_iou(box1,
             box2,
             x1y1x2y2=True,
             GIoU=False,
             DIoU=False,
             CIoU=False,
             eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4xn, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            # center distance squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter
                    )  # iou = inter / (area1 + area2 - inter)


def box_candidates(box1,
                   box2,
                   wh_thr=2,
                   ar_thr=20,
                   area_thr=0.1):  # box1(4,n), box2(4,n)
    """Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 /
                                            (w1 * h1 + 1e-16) > area_thr) & (
                                                ar < ar_thr)  # candidates


def clip_coords(boxes: Tensor, img_shape: Tuple[int, int]) -> None:
    """Clip bounding xyxy bounding boxes to image shape

    Args:
        boxes: tensor with shape Nx4 (x1,y1,x2,y2)
        img_shape: image size tuple, (height, width)
    """
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape: Tuple[int, int],
                 coords: torch.Tensor,
                 img0_shape: Tuple[int, int],
                 ratio_pad: Optional[Tuple[Tuple[float, float],
                                           Tuple[float, float]]] = None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] = coords[:, :4] / gain
    clip_coords(coords, img0_shape)
    return coords
