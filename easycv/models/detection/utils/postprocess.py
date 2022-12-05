from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from easycv.models.detection.utils import box_cxcywh_to_xyxy


class DetrPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self,
                 num_select=None,
                 use_centerness=False,
                 use_iouaware=False) -> None:
        super().__init__()
        self.num_select = num_select
        self.use_centerness = use_centerness
        self.use_iouaware = use_iouaware

    @torch.no_grad()
    def forward(self, outputs, target_sizes, img_metas):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        if self.num_select is None:
            prob = F.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1)
            boxes = box_cxcywh_to_xyxy(out_bbox)

            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h],
                                    dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]
        else:
            if self.use_centerness and self.use_iouaware:
                prob = out_logits.sigmoid(
                )**0.45 * outputs['pred_centers'].sigmoid(
                )**0.05 * outputs['pred_ious'].sigmoid()**0.5
            elif self.use_centerness:
                prob = out_logits.sigmoid() * outputs['pred_centers'].sigmoid()
            elif self.use_iouaware:
                prob = out_logits.sigmoid() * outputs['pred_ious'].sigmoid()
            else:
                prob = out_logits.sigmoid()

            topk_values, topk_indexes = torch.topk(
                prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1,
                                 topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h],
                                    dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = {
            'detection_boxes': [boxes[0].cpu().numpy()],
            'detection_scores': [scores[0].cpu().numpy()],
            'detection_classes': [labels[0].cpu().numpy().astype(np.int32)],
            'img_metas': img_metas
        }

        return results


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


# refer to easycv/models/detection/detectors/yolox/postprocess.py and test.py to rebuild a torch-blade-trtplugin NMS, which is checked by zhoulou in test.py
# infer docker images is : registry.cn-shanghai.aliyuncs.com/pai-ai-test/eas-service:easycv_blade_181_export


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.numel():
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >=
                     conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.numel():
            continue

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.8.0'):
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4], detections[:, 4] * detections[:, 5],
                detections[:, 6], nms_thre)
        else:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4], detections[:, 4] * detections[:, 5],
                nms_thre)

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output
