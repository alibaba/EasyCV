# Export tutorial

We support the following ways to export models.

**Original**

Original model saves the state dict of model. One should build model in advance and then load the model state dict.

**torch.jit**

Torch.jit is used to save the TorchScript model. It can be used independently from Python. It is convenient to be deployed in various environments and has little dependency on hardware. It can also reduce the inference time. For more details, you can refer to the official tutorial: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html

**Blade**

Blade Model is used to greatly accelerate the inference process. It combines the technology of computational graph optimization, TensorRT/oneDNN,  AI compiler optimization, etc. For more details, you can refer to the official tutorial: https://help.aliyun.com/document_detail/205129.html

**End2end**

End2end model wraps the preprocess and postprocess process along with the model. Therefore, given an input image, the model can be directly used for inference. 



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

#### Original model

Eport the orginal model by setting the export config as:

```shell
export = dict(use_jit=False, export_blade=False, end2end=False)
```

#### Script model

Eport the script model by setting the export config as:

```shell
export = dict(use_jit=True, export_blade=False, end2end=False)
```

#### Blade model

Eport the blade model by setting the export config as:

```shell
export = dict(use_jit=True, export_blade=True, end2end=False)
```

You can choose not to save the jit model by setting use_jit=False.

The blade environment must be installed successfully to export a blade model.

To install the blade, you can refer to https://help.aliyun.com/document_detail/205134.html.

#### End2end model

Eport the model in the end2end mode by setting ''end2end=True'' in the export config:

```shell
export = dict(use_jit=True, export_blade=True, end2end=True)
```

You should define your own preprocess and postprocess as below (please refer to: https://pytorch.org/docs/stable/jit.html?highlight=jit#module-torch.jit ) or the default test pipeline will be used.

```python
@torch.jit.script
def preprocess_fn(image, traget_size=(640, 640)):
		"""Process the data input to model."""
    pass

@torch.jit.script
def postprocess_fn(output):
		"""Process output values of the model."""
    pass

# define your own export wrapper
End2endModelExportWrapper(
    model,
    preprocess_fn=preprocess_fn,
    postprocess_fn=postprocess_fn)
```



### Inference with the Exported Model

#### Non-End2end model

```python
image_path = 'data/demo.jpg'
input_data_list =[np.asarray(Image.open(image_path))]

# define the preprocess function
test_pipeline = [
    dict(type='MMResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MMPad', pad_to_square=True, pad_val=(114.0, 114.0, 114.0)),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]

def preprocess(img):
  	pipeline = [build_from_cfg(p, PIPELINES) for p in test_pipeline]
    transform = Compose(pipeline)
    return transform(img)['img']


with io.open(jit_model_path, 'rb') as infile:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.jit.load(infile, device)

    for idx, img in enumerate(input_data_list):
        if type(img) is not np.ndarray:
            img = np.asarray(img)
        img = preprocess(img)
        output = model(img)
        output = postprocess(output)
        print(output)
```

#### End2end model


```python
image_path = 'data/demo.jpg'
input_data_list =[np.asarray(Image.open(image_path))]

with io.open(jit_model_path, 'rb') as infile:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.jit.load(infile, device)

    for idx, img in enumerate(input_data_list):
        if type(img) is not np.ndarray:
            img = np.asarray(img)
        img = torch.from_numpy(img).to(device)
        output = model(img)
        print(output)
```



### Inference Time Comparisons

Use the YOLOX-S model as an example, the inference process can be greatly accelerated by using the script and blade model.

|  Model  |       Mode       |  FPS   |
| :-----: | :--------------: | :----: |
| YOLOX-S |     Original     | 54.02  |
| YOLOX-S |      Script      | 89.33  |
| YOLOX-S |      Blade       | 174.38 |
| YOLOX-S | Script (End2End) | 86.62  |
| YOLOX-S | Blade (End2End)  | 160.86 |
