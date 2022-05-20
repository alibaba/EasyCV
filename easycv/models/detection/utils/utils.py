# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from typing import List
# from easycv.utils.bbox_util import xywh2xyxy

def output_postprocess(outputs, img_metas=None):
    detection_boxes = []
    detection_scores = []
    detection_classes = []
    img_metas_list = []

    for i in range(len(outputs)):
        if img_metas:
            img_metas_list.append(img_metas[i])
        if outputs[i] is not None:
            bboxes = outputs[i][:, 0:4] if outputs[i] is not None else None
            if img_metas:
                bboxes /= img_metas[i]['scale_factor'][0]
            detection_boxes.append(bboxes.cpu().numpy())
            detection_scores.append(
                (outputs[i][:, 4] * outputs[i][:, 5]).cpu().numpy())
            detection_classes.append(outputs[i][:, 6].cpu().numpy().astype(
                np.int32))
        else:
            detection_boxes.append(None)
            detection_scores.append(None)
            detection_classes.append(None)

    test_outputs = {
        'detection_boxes': detection_boxes,
        'detection_scores': detection_scores,
        'detection_classes': detection_classes,
        'img_metas': img_metas_list
    }

    return test_outputs


def non_max_suppression_jit(prediction,
                            batch_size: int = 1,
                            conf_thres: float = 0.1,
                            iou_thres: float = 0.6,
                            agnostic: bool = False) -> List[torch.Tensor]:
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    # nc = prediction[0].shape[1] - 5  # number of classes
    # xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 100  # maximum number of detections per image
    redundant = True  # require redundant detections
    # multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    multi_label = False
    # output = []
    output = [
        torch.zeros((max_det, prediction[0].shape[1]))
        for i in range(batch_size)
    ]
    for xi in range(batch_size):  # image index, image inference
        x = prediction[xi]
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # x = x[xc[xi]]  # confidence

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()),
                          1)[conf.view(-1) > conf_thres]

        # Batched NMS
        # c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:,
                                        4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)

        output[xi] = x[i]

    return output
