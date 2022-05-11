# Use mmlab's models in EasyCV

**We only support mmdet's models and do not support other series in mmlab and other modules such as transforms, dataset api, etc. are not supported either.**

The models module of EasyCV is divided into four modules: `backbone`, `head`, `neck`, and `model`.

So we support the models combination of EasyCV and mmlab from these four levels.

**We will not adapt the other apis involved in these four levels modules, we package the entire api for use.**

> **Note: **
>
> **If you want to combine the models part of mmdet and easycv, please pay attention to the compatibility between the apis, we do not guarantee that the api of EasyCV and mmlab are compatible.**

Take the `MaskRCNN` model as an example, please refer to [mask_rcnn_r50_fpn.py](https://github.com/alibaba/EasyCV/tree/master/configs/detection/mask_rcnn/mask_rcnn_r50_fpn.py). Except for the backbone, other parts in this model are all mmdet apis.

The framework of `MaskRCNN` can be divided into the following parts from the `backbone`, `head`, `neck`, and `model` levels：

- backbone: `ResNet`

- head：`RPNHead`,	`StandardRoIHead`

- neck: `FPN`

- model: `MaskRCNN`

The configuration adapt for mmdet is as follows:

```python
mmlab_modules = [
    dict(type='mmdet', name='MaskRCNN', module='model'),
    # dict(type='mmdet', name='ResNet', module='backbone'), # comment out, use EasyCV ResNet
    dict(type='mmdet', name='FPN', module='neck'),
    dict(type='mmdet', name='RPNHead', module='head'),
    dict(type='mmdet', name='StandardRoIHead', module='head'),
]
```

> Parameters:
>
> 	- type: the name of the open source, only `mmdet` is supported
> 	- name:  the name of api
> 	- Module: The name of the module to which the api belongs, only  `backbone`,`head`,`neck`,`model` are supported.

In this configuration , the `head`, `neck`, and `model` parts specify the type as `mmdet`, except for `backbone`.

**No configured api will use the EasyCV api by default, , such as backbone (ResNet).**

**For other explicitly configured type as `mmdet`, we will use the mmdet api.**

Which is:

- `MaskRCNN`(model): Use mmdet's `MaskRCNN` api.

- `ResNet`(backbone): Use EasyCV's `ResNet` api.

  > Note that the parameters of the `ResNet`of mmdet and EasyCV are different. Please pay attention to it!.

- `RPNHead`(head): Use mmdet's `RPNHead` api.

  > Note that all the other apis configured in `RPNHead`, such as `AnchorGenerator`, `DeltaXYWHBBoxCoder`, etc., are all mmdet's apis, because we package the entire api for use.

- `StandardRoIHead`(head): Use mmdet's `StandardRoIHead` api.

  > Note that all the other apis configured in `StandardRoIHead`, such as `SingleRoIExtractor`, `SingleRoIExtractor`, etc., are all mmdet's apis, because we package the entire api for use.

- `FPN`(neck): Use mmdet's `FPN` api.
