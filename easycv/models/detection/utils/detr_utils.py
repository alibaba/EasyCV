import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.detection.utils import box_cxcywh_to_xyxy


class DetrPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=None) -> None:
        super().__init__()
        self.num_select = num_select

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
