# Export tutorial

We support the following ways to export YOLOX-PAI models.

**Original**

Original model saves the state dict of model. One should export model in advance to infer an image.

**Torch.jit**

Torch.jit is used to save the TorchScript model. It can be used independently from Python. It is convenient to be deployed in various environments and has little dependency on hardware. It can also reduce the inference time. For more details, you can refer to the official tutorial: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html

**Blade**

Blade Model is used to greatly accelerate the inference process. It combines the technology of computational graph optimization, TensorRT/oneDNN,  AI compiler optimization, etc. For more details, you can refer to the official tutorial: https://help.aliyun.com/document_detail/205129.html

**End2end**

To simplify and accelerate the end2end inference, we support to wrap the preprocess/postprocess process, respectively.

You can choose to export model with or without preprocess/postprocess by setting different configs.

### Installation
You should install a blade environment first to use blade optimization.
See [link](https://help.aliyun.com/document_detail/205134.html.) for instruction.

You are also recommended to use our provided docker image.
```shell
sudo docker pull registry.cn-shanghai.aliyuncs.com/pai-ai-test/eas-service:blade_cu111_easycv
```
### Export model

```shell
python tools/export.py \
		${CONFIG_PATH} \
		${CHECKPOINT} \
		${EXPORT_PATH}
```

<details>
<summary>Arguments</summary>


- `CONFIG_PATH`: the config file path of a detection method
- `CHECKPOINT`:your checkpoint file of a detection method named as epoch_*.pth.
- `EXPORT_PATH`: your path to save export model

</details>

**Examples:**

```shell
python tools/export.py configs/detection/yolox/yolox_s_8xb16_300e_coco.py \
        work_dirs/detection/yolox/epoch_300.pth \
        work_dirs/detection/yolox/epoch_300_export.pth
```

**Export configs:**
```shell
# default
export = dict(export_type='raw',              # exported model type ['raw','jit','blade']
              preprocess_jit=True,            # whether to save a preprocess jit model
              static_opt=True,                # whether to use static shape to optimize model
              batch_size=1,                   # batch_size if the static shape
              blade_config=dict(
                  enable_fp16=True,
                  fp16_fallback_op_ratio=0.05 # fallback to fp32 ratio for blade optimize
                                              # the difference between the fp16 and fp32 results of all layers will be computed
                                              # The layers with larger difference are likely to fallback to fp16
                                              # if the optimized result is not ture, you can choose a larger ratio.
              ),
              use_trt_efficientnms=False)      # whether to wrap the trt_nms into model
```

We allow users to use different combinations of "export_type", "preprocess_jit", and "use_trt_efficientnms" as shown in the Table below to export the model.

### Inference
Take jit script model as an example. Set the config file as below:
```shell
export = dict(export_type='jit',
              preprocess_jit=True,
              static_opt=True,
              batch_size=1,
              use_trt_efficientnms=False)
```

Then, you can obtain the following exported model:
``` shell
yolox_s.pt.jit
yolox_s.pt.jit.config.json
yolox_s.pt.preprocess (only exists when set preprocess_jit = True)
```
You can simply use our EasyCV predictor to use the exported model:
```python
import cv2
from easycv.predictors import TorchYoloXPredictor

output_ckpt = 'yolox_s.pt.jit'
detector = TorchYoloXPredictor(output_ckpt,use_trt_efficientnms=False)

img = cv2.imread('000000017627.jpg')
output = detector.predict([img])
print(output)
```
We highly recommend you to use EasyCV predictor for inference with different export types below. Use YOLOX-s as an example, we test the en2end inference time of different models on a single NVIDIA Tesla V100.


| export_type | preprocess_jit | use_trt_efficientnms | Infer time (end2end) /ms |
| :---------: | :------------: | :------------------: | :----------------------: |
|     ori     |       -        |          -           |          24.58           |
|     jit     |     False      |        False         |          18.30           |
|     jit     |     False      |         True         |          18.38           |
|     jit     |      True      |        False         |          13.44           |
|     jit     |      True      |         True         |          13.04           |
|    blade    |     False      |        False         |           8.72           |
|    blade    |     False      |         True         |           9.39           |
|    blade    |      True      |        False         |           3.93           |
|    blade    |      True      |         True         |           4.53           |


Or you can use our exported model with a simple environment to deploy our model on your own device:
```python
import io
import torch
import cv2
import numpy as np
import torchvision

# load img
img = cv2.imread('test.jpg')
img = torch.tensor(img).unsqueeze(0).cuda()

# load model
model_path = 'yolox_s.pt.jit'
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
print(outputs[0].shape)

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
```

Note that we only allow to export an end2end TorchScript Model. For the exported Blade model, NMS is not allowed to be wrapped into the model. You should follow [postprocess.py](https://github.com/alibaba/EasyCV/tree/master/easycv/models/detection/utils/postprocess.py) to add the postprocess procedure.
