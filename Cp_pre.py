import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as t_f
from torchvision.transforms import ToPILImage

def preprocess(image):
    input_h, input_w = (640, 640)
    image = image.permute(2, 0, 1)
    image = image[[2, 1, 0], :, :]
    image = torch.unsqueeze(image, 0)
    ori_h, ori_w = image.shape[-2:]

    mean = np.float64(np.array([123.675, 116.28, 103.53]))
    std = np.float64(np.array([58.395, 57.12, 57.375]))

    keep_ratio = True
    if not keep_ratio:
        out_image = t_f.resize(image, [input_h, input_w])
        out_image = t_f.normalize(out_image, mean, std)
        pad_l, pad_t, scale = 0, 0, 1.0
    else:
        scale = min(input_h / ori_h, input_w / ori_w)
        resize_h, resize_w = int(ori_h * scale), int(ori_w * scale)
        pad_h, pad_w = input_h - resize_h, input_w - resize_w
        pad_l, pad_t = 0, 0
        pad_r, pad_b = pad_w - pad_l, pad_h - pad_t
        out_image = t_f.resize(image, [resize_h, resize_w])
        out_image = t_f.pad(
            out_image, [pad_l, pad_t, pad_r, pad_b], fill=114)
        out_image = t_f.normalize(out_image, mean, std)

    h, w = out_image.shape[-2:]
    output_info = {
        'pad': (float(pad_l), float(pad_t)),
        'scale_factor': (float(scale), float(scale)),
        'ori_img_shape': (float(ori_h), float(ori_w)),
        'img_shape': (float(h), float(w))
    }
    return out_image, output_info

BASE_LOCAL_PATH = os.path.expanduser('~/easycv_nfs/')
DET_DATA_SMALL_COCO_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                         'data/detection/small_coco')

img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000037777.jpg')

img = np.asarray(Image.open(img))

img = torch.from_numpy(img).cuda()

out = preprocess(img)

print(out)