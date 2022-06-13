# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from torch.autograd import Function
from torch.nn import functional as F


class SigmoidGeometricMean(Function):
    """Forward and backward function of geometric mean of two sigmoid
    functions.

    This implementation with analytical gradient function substitutes
    the autograd function of (x.sigmoid() * y.sigmoid()).sqrt(). The
    original implementation incurs none during gradient backprapagation
    if both x and y are very small values.
    """

    @staticmethod
    def forward(ctx, x, y):
        x_sigmoid = x.sigmoid()
        y_sigmoid = y.sigmoid()
        z = (x_sigmoid * y_sigmoid).sqrt()
        ctx.save_for_backward(x_sigmoid, y_sigmoid, z)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x_sigmoid, y_sigmoid, z = ctx.saved_tensors
        grad_x = grad_output * z * (1 - x_sigmoid) / 2
        grad_y = grad_output * z * (1 - y_sigmoid) / 2
        return grad_x, grad_y


sigmoid_geometric_mean = SigmoidGeometricMean.apply


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
