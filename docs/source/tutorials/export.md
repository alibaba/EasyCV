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
              use_trt_efficientnms=True)      # whether to wrap the trt_nms into model
```

### Inference Time Comparisons
Use YOLOX-s as an example, we test the en2end inference time of models exported with different configs.
Note that blade optimization needs warmup, and we report average time among 1000 experiments on a single NVIDIA Tesla V100.


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
