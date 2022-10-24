import io
import torch
import cv2
import numpy as np
import torchvision

# load img
img = cv2.imread('/home/zouxinyi.zxy/easycv_nfs/data/detection/small_coco/val2017/000000522713.jpg')
img = torch.tensor(img).unsqueeze(0).cuda()

# load model
model_path = '/home/zouxinyi.zxy/easycv_nfs/pretrained_models/detection/infer_yolox/epoch_300_pre_notrt_e2e.pt.jit'
preprocess_path = '.'.join(
    model_path.split('.')[:-1] + ['preprocess'])
with io.open(preprocess_path, 'rb') as infile:
    preprocess = torch.jit.load(infile)
with io.open(model_path, 'rb') as infile:
    model = torch.jit.load(infile)

# preporcess with the exported model or use your own preprocess func
img, img_info = preprocess(img)

# forward with nms [b,c,h,w] -> List[[n,7]]
# n means the predicted box num of each img
# 7 means [x1,y1,x2,y2,obj_conf,cls_conf,cls]
outputs = model(img)

# postprocess the output information into dict or your own data structure
# slice box,score,class & rescale box
detection_boxes = []
detection_scores = []
detection_classes = []
bboxes = outputs[0][:, 0:4]
bboxes /= img_info['scale_factor'][0]
detection_boxes.append(bboxes.cpu().detach().numpy())
detection_scores.append(
    (outputs[0][:, 4] * outputs[0][:, 5]).cpu().detach().numpy())
detection_classes.append(outputs[0][:, 6].cpu().detach().numpy().astype(
    np.int32))

final_outputs = {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
        }

print(final_outputs)

