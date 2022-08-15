# from easycv.models.detection.detectors.yolox import YOLOX
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose

from easycv.datasets.registry import PIPELINES
from easycv.models import build_model
from easycv.models.detection.detectors.yolox.postprocess import \
    create_tensorrt_postprocess
from easycv.models.detection.utils import postprocess
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.registry import build_from_cfg

if __name__ == '__main__':
    #a = YOLOX(decode_in_inference=False).eval()
    cfg = sys.argv[1]
    ckpt_path = sys.argv[2]

    cfg = mmcv_config_fromfile(cfg)
    model = build_model(cfg.model)
    load_checkpoint(model, ckpt_path, map_location='cpu')
    model = model.eval()

    test_pipeline = cfg.test_pipeline
    CLASSES = cfg.CLASSES

    pipeline = [build_from_cfg(p, PIPELINES) for p in test_pipeline]
    pipeline = Compose(pipeline)

    # 8400 ishard code, need to reimplement to  sum(img_w / stride_i + img_h /stride_i)
    example_scores = torch.randn([1, 8400, 85], dtype=torch.float32)
    trt_ext = create_tensorrt_postprocess(
        example_scores, iou_thres=model.nms_thre, score_thres=model.test_conf)

    # img_path = '/apsara/xinyi.zxy/data/coco/val2017/000000129062.jpg'
    img_path = '/apsara/xinyi.zxy/data/coco/val2017/000000254016.jpg'
    # img = cv2.imread(img_path)
    img = Image.open(img_path)
    if type(img) is not np.ndarray:
        img = np.asarray(img)

    # ori_img_shape = img.shape[:2]
    data_dict = {'img': img}
    data_dict = pipeline(data_dict)
    img = data_dict['img']
    img = torch.unsqueeze(img._data, 0)
    # print(img.shape)
    model.decode_in_inference = False
    # print(type(model), model.decode_in_inference)
    c = model.forward_export(img)

    # print(type(c), c.shape)
    print(model.test_conf, model.nms_thre, model.num_classes,
          model.decode_in_inference)
    tc = model.head.decode_outputs(c, c[0].type())
    # print(type(tc))
    # print(tc.shape)

    import copy
    tcback = copy.deepcopy(tc)

    tpa = postprocess(tc, model.num_classes, model.test_conf,
                      model.nms_thre)[0]
    # print(tpa)
    tpa[:, 4] = tpa[:, 4] * tpa[:, 5]
    tpa[:, 5] = tpa[:, 6]
    tpa = tpa[:, :6]
    # print("fuck tpa:", len(tpa), tpa[0].shape)
    box_a = tpa[:, :4]
    score_a = tpa[:, 4]
    id_a = tpa[:, 5]
    # print(tpa)

    # trt_ext must be cuda
    tcback = tcback
    tpb = trt_ext.forward(tcback)
    # print("fuck tpb:",len(tpb))

    valid_length = min(len(tpa), tpb[2].shape[1])
    print(valid_length)
    valid_length = min(valid_length, 30)

    box_a = box_a[:valid_length]
    score_a = score_a[:valid_length]
    id_a = id_a[:valid_length]

    print(tpb[1].shape)
    print(tpb[2].shape)
    print(tpb[3].shape)

    box_b = tpb[1][:, :valid_length, :].cpu().view(box_a.shape)
    score_b = tpb[2][:, :valid_length].cpu().view(score_a.shape)
    id_b = tpb[3][:, :valid_length].cpu().view(id_a.shape)

    def get_diff(input_a, input_b, name='score'):
        print('name:', name)
        print('shape:', input_a.shape)
        print('max_diff  :', torch.max(input_a - input_b))
        print('avg_diff  :', torch.mean(input_a - input_b))
        print('totol_diff:', torch.sum(input_a - input_b))

    get_diff(box_a, box_b, 'box')
    get_diff(score_a, score_b, 'score')
    get_diff(id_a, id_a, 'id')

    if 0:
        from easycv.predictors import TorchYoloXPredictor
        img = Image.open(img_path)
        pred = TorchYoloXPredictor('models/predict.pt')
        m = pred.predict([img])
        print(m)
