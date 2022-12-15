# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
import random
import seaborn as sns

import mmcv

def reid_predictor(detection_results, reid_model):
    pred_dets = detection_results['boxes']  # cls_id, score, x0, y0, x1, y1
    pred_xyxys = pred_dets[:, 2:6]

    ori_image = detection_results['ori_image']
    ori_image_shape = ori_image.shape[:2]
    pred_xyxys, keep_idx = clip_box(pred_xyxys, ori_image_shape)

    if len(keep_idx[0]) == 0:
        return pred_embeddings

    pred_dets = pred_dets[keep_idx[0]]
    pred_xyxys = pred_dets[:, 2:6]

    w, h = self.tracker.input_size
    crops = get_crops(pred_xyxys, ori_image, w, h)

    # to keep fast speed, only use topk crops
    # crops = crops[:50]  # reid_batch_size

    pred_embeddings = reid_model(crops)

    return pred_embeddings

def clip_box(xyxy, ori_image_shape):
    H, W = ori_image_shape
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], a_min=0, a_max=W)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], a_min=0, a_max=H)
    w = xyxy[:, 2:3] - xyxy[:, 0:1]
    h = xyxy[:, 3:4] - xyxy[:, 1:2]
    mask = np.logical_and(h > 0, w > 0)
    keep_idx = np.nonzero(mask)
    return xyxy[keep_idx[0]], keep_idx

def get_crops(xyxy, ori_img, w, h):
    crops = []
    xyxy = xyxy.astype(np.int64)
    ori_img = ori_img.transpose(1, 0, 2)  # [h,w,3]->[w,h,3]
    for i, bbox in enumerate(xyxy):
        crop = ori_img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        crops.append(crop)
    crops = preprocess_reid(crops, w, h)
    return crops

def detection_result_filter(bboxes, scores, classes, target_classes, target_thresholds=None):
    # post process to filter result
    bboxes_tmp = []
    scores_tmp = []
    classes_tmp = []
    assert len(target_classes)==len(target_thresholds), "detection post process, class filter need target_classes and target_thresholds both, and should be same length!"

    for bidx, bcls in enumerate(classes):
        if bcls in target_classes and scores[bidx] > target_thresholds[target_classes.index(bcls)]:
            bboxes_tmp.append(bboxes[bidx])
            scores_tmp.append(scores[bidx])
            classes_tmp.append(classes[bidx])
    bboxes = np.array(bboxes_tmp)
    scores = np.array(scores_tmp)
    classes = np.array(classes_tmp)
    return bboxes, scores, classes

def results2outs(bbox_results=None,
                 **kwargs):
    """Restore the results (list of results of each category) into the results
    of the model forward.

    Args:
        bbox_results (list[np.ndarray]): Each list denotes bboxes of one
            category.

    Returns:
        tuple: tracking results of each class. It may contain keys as belows:

        - bboxes (np.ndarray): shape (n, 5)
        - ids (np.ndarray): shape (n, )
    """
    outputs = dict()

    if len(bbox_results) > 0:

        bboxes = bbox_results
        if bboxes.shape[1] == 5:
            outputs['bboxes'] = bboxes
        elif bboxes.shape[1] == 6:
            ids = bboxes[:, 0].astype(np.int64)
            bboxes = bboxes[:, 1:]
            outputs['bboxes'] = bboxes
            outputs['ids'] = ids
        else:
            raise NotImplementedError(
                f'Not supported bbox shape: (N, {bboxes.shape[1]})')

    return outputs

def random_color(seed):
    """Random a color according to the input seed."""
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color

def imshow_tracks(img,
                bboxes,
                ids,
                classes=None,
                score_thr=0.0,
                thickness=2,
                font_scale=0.4,
                show=False,
                wait_time=0,
                out_file=None):
    """Show the tracks with opencv."""
    if isinstance(img, str):
        img = mmcv.imread(img)
    if bboxes is not None and ids is not None:
        assert bboxes.ndim == 2
        assert ids.ndim == 1
        assert bboxes.shape[1] == 5

        img_shape = img.shape
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

        inds = np.where(bboxes[:, -1] > score_thr)[0]
        bboxes = bboxes[inds]
        ids = ids[inds]

        text_width, text_height = 9, 13
        for i, (bbox, id) in enumerate(zip(bboxes, ids)):
            x1, y1, x2, y2 = bbox[:4].astype(np.int32)
            score = float(bbox[-1])

            # bbox
            bbox_color = random_color(id)
            bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

            # score
            text = '{:.02f}'.format(score)
            width = len(text) * text_width
            img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
            cv2.putText(
                img,
                text, (x1, y1 + text_height - 2),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                color=(0, 0, 0))

            # id
            text = str(id)
            width = len(text) * text_width
            img[y1 + text_height:y1 + 2 * text_height,
                x1:x1 + width, :] = bbox_color
            cv2.putText(
                img,
                str(id), (x1, y1 + 2 * text_height - 2),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                color=(0, 0, 0))

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img

def show_result(
                img,
                result,
                score_thr=0.0,
                thickness=1,
                font_scale=0.5,
                show=False,
                out_file=None,
                wait_time=0,
                **kwargs):
    """Visualize tracking results.

    Args:
        img (str | ndarray): Filename of loaded image.
        result (dict): Tracking result.
            - The value of key 'track_bboxes' is list with length
            num_classes, and each element in list is ndarray with
            shape(n, 6) in [id, tl_x, tl_y, br_x, br_y, score] format.
            - The value of key 'det_bboxes' is list with length
            num_classes, and each element in list is ndarray with
            shape(n, 5) in [tl_x, tl_y, br_x, br_y, score] format.
        thickness (int, optional): Thickness of lines. Defaults to 1.
        font_scale (float, optional): Font scales of texts. Defaults
            to 0.5.
        show (bool, optional): Whether show the visualizations on the
            fly. Defaults to False.
        out_file (str | None, optional): Output filename. Defaults to None.

    Returns:
        ndarray: Visualized image.
    """
    assert isinstance(result, dict)
    track_bboxes = result.get('track_bboxes', None)
    if isinstance(img, str):
        img = mmcv.imread(img)
    outs_track = results2outs(bbox_results=track_bboxes)
    img = imshow_tracks(
        img,
        outs_track.get('bboxes', None),
        outs_track.get('ids', None),
        score_thr=score_thr,
        thickness=thickness,
        font_scale=font_scale,
        show=show,
        out_file=out_file,
        wait_time=wait_time)
    return img