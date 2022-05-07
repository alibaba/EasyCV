# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/visualization/image.py
import math
import os
from os.path import dirname as opd

import cv2
import mmcv
import numpy as np
from mmcv.utils.misc import deprecated_api_warning
from PIL import Image, ImageDraw, ImageFont


def get_font_path():
    root_path = opd(opd(opd(os.path.realpath(__file__))))
    # find in whl
    find_path_whl = os.path.join(root_path, 'resource/simhei.ttf')
    # find in source code
    find_path_source = os.path.join(opd(root_path), 'resource/simhei.ttf')
    if os.path.exists(find_path_whl):
        return find_path_whl
    elif os.path.exists(find_path_source):
        return find_path_source
    else:
        raise ValueError('Not find font file both in %s and %s' %
                         (find_path_whl, find_path_source))


_FONT_PATH = get_font_path()


def put_text(img, xy, text, fill, size=20):
    """support chinese text
    """
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(_FONT_PATH, size=size, encoding='utf-8')
    draw.text(xy, text, fill=fill, font=fontText)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


def imshow_label(img,
                 labels,
                 text_color='blue',
                 font_size=20,
                 thickness=1,
                 font_scale=0.5,
                 intervel=5,
                 show=True,
                 win_name='',
                 wait_time=0,
                 out_file=None):
    """Draw images with labels on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        labels (str or list[str]): labels of each image.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        font_size (int): Size of font.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        intervel（int): interval pixels between multiple labels
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    img = mmcv.imread(img)
    img = np.ascontiguousarray(img)
    labels = [labels] if isinstance(labels, str) else labels

    cur_height = 0
    for label in labels:
        # roughly estimate the proper font size
        text_size, text_baseline = cv2.getTextSize(label,
                                                   cv2.FONT_HERSHEY_DUPLEX,
                                                   font_scale, thickness)

        org = (text_baseline + text_size[1],
               text_baseline + text_size[1] + cur_height)

        # support chinese text
        # TODO: Unify the font of cv2 and PIL, and auto get font_size according to the font_scale
        img = put_text(img, org, text=label, fill=text_color, size=font_size)

        # cv2.putText(img, label, org, cv2.FONT_HERSHEY_DUPLEX, font_scale,
        #             mmcv.color_val(text_color), thickness)

        cur_height += text_baseline + text_size[1] + intervel

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def imshow_bboxes(img,
                  bboxes,
                  labels=None,
                  colors='green',
                  text_color='white',
                  font_size=20,
                  thickness=1,
                  font_scale=0.5,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes with labels (optional) on an image. This is a wrapper of
    mmcv.imshow_bboxes.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): ndarray of shape (k, 4), each row is a bbox in
            format [x1, y1, x2, y2].
        labels (str or list[str], optional): labels of each bbox.
        colors (list[str or tuple or :obj:`Color`]): A list of colors.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        font_size (int): Size of font.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """

    # adapt to mmcv.imshow_bboxes input format
    bboxes = np.split(
        bboxes, bboxes.shape[0], axis=0) if bboxes.shape[0] > 0 else []
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [mmcv.color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    img = mmcv.imshow_bboxes(
        img,
        bboxes,
        colors,
        top_k=-1,
        thickness=thickness,
        show=False,
        out_file=None)

    if labels is not None:
        assert len(labels) == len(bboxes)

        for bbox, label, color in zip(bboxes, labels, colors):
            label = str(label)
            bbox_int = bbox[0, :4].astype(np.int32)
            # roughly estimate the proper font size
            text_size, text_baseline = cv2.getTextSize(label,
                                                       cv2.FONT_HERSHEY_DUPLEX,
                                                       font_scale, thickness)
            text_x1 = bbox_int[0]
            text_y1 = max(0, bbox_int[1] - text_size[1] - text_baseline)
            text_x2 = bbox_int[0] + text_size[0]
            text_y2 = text_y1 + text_size[1] + text_baseline
            cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color,
                          cv2.FILLED)
            # cv2.putText(img, label, (text_x1, text_y2 - text_baseline),
            #             cv2.FONT_HERSHEY_DUPLEX, font_scale,
            #             mmcv.color_val(text_color), thickness)

            # support chinese text
            # TODO: Unify the font of cv2 and PIL, and auto get font_size according to the font_scale
            img = put_text(
                img, (text_x1, text_y1),
                text=label,
                fill=text_color,
                size=font_size)

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    return img


@deprecated_api_warning({'pose_limb_color': 'pose_link_color'})
def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    img = mmcv.imread(img)
    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)
            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                if kpt_score > kpt_score_thr:
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                   radius, (int(r), int(g), int(b)), -1)
                        transparency = max(0, min(1, kpt_score))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                                   (int(r), int(g), int(b)), -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)
            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                        and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                        and pos2[1] > 0 and pos2[1] < img_h
                        and kpts[sk[0], 2] > kpt_score_thr
                        and kpts[sk[1], 2] > kpt_score_thr):
                    r, g, b = pose_link_color[sk_id]
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        X = (pos1[0], pos2[0])
                        Y = (pos1[1], pos2[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                        angle = math.degrees(
                            math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = 2
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)),
                            (int(length / 2), int(stickwidth)), int(angle), 0,
                            360, 1)
                        cv2.fillConvexPoly(img_copy, polygon,
                                           (int(r), int(g), int(b)))
                        transparency = max(
                            0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.line(
                            img,
                            pos1,
                            pos2, (int(r), int(g), int(b)),
                            thickness=thickness)

    return img
