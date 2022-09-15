# This is a TensorRT Plugin Python Wrapper Link implementation, original plugin documents refers to
# https://github.com/NVIDIA/TensorRT/tree/main/plugin/
# We use python wrapper to build ONNX-TRTPlugin Engine and then wrapper as a jit script module, after this,
# we could replace some original model's OP with this plugin during Blade Export to speed up those are not
# well optimized by original Blade
# Here we provide a TRTPlugin-EfficientNMS implementation

import torch
from torch import nn


class TRT8_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(
            0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version='1',
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_TRT8(nn.Module):
    '''onnx module with TensorRT NMS operation.'''

    def __init__(self,
                 max_obj=100,
                 iou_thres=0.45,
                 score_thres=0.25,
                 max_wh=None,
                 device=None):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, x):
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        num_det, det_boxes, det_scores, det_classes = TRT8_NMS.apply(
            box, score, self.background_class, self.box_coding,
            self.iou_threshold, self.max_obj, self.plugin_version,
            self.score_activation, self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


def create_tensorrt_efficientnms(example_scores,
                                 iou_thres=0.45,
                                 score_thres=0.25):
    """

    """
    from torch_blade import tensorrt
    import torch_blade._torch_blade._backends as backends
    import io

    model = torch.jit.trace(
        ONNX_TRT8(iou_thres=iou_thres, score_thres=score_thres),
        example_scores)
    example_outputs = model(example_scores)

    input_names = ['input']
    output_names = [
        'num_det', 'detection_boxes', 'detection_scores', 'detection_classes'
    ]
    with io.BytesIO() as onnx_proto_f:
        torch.onnx.export(
            model,
            example_scores,
            onnx_proto_f,
            input_names=input_names,
            output_names=output_names,
            example_outputs=example_outputs)
        onnx_proto = onnx_proto_f.getvalue()

    def _copy_meta(data, name, dtype, sizes):
        data.name = name
        if dtype.is_floating_point:
            data.dtype = 'Float'
        else:
            data.dtype = 'Int'
        data.sizes = sizes
        return data

    state = backends.EngineState()
    state.inputs = [
        _copy_meta(backends.TensorInfo(), name, tensor.dtype,
                   list(tensor.shape))
        for name, tensor in zip(input_names, [example_scores])
    ]
    state.outputs = [
        _copy_meta(backends.TensorInfo(), name, tensor.dtype, [])
        for name, tensor in zip(output_names, example_outputs)
    ]
    state = tensorrt.cvt_onnx_to_tensorrt(onnx_proto, state, [], dict())

    class Model(torch.nn.Module):

        def __init__(self, state):
            super().__init__()
            self._trt_engine_ext = backends.create_engine(state)

        def forward(self, x):
            return self._trt_engine_ext.execute([x])

    trt_ext = torch.jit.script(Model(state))
    return trt_ext
