# Export tutorial

We support three kinds of the export model, the original model, the script model, and the blade model. Script (Jit) and Blade are used to accelerate the inference process. We also support the end2end export mode to wrapper the preprocess and postprocess  with the model.

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

You should define your own preprocess and postprocess as below or the default test pipeline will be used.

```python
@torch.jit.script
class PreProcess:
    """Process the data input to model."""
		def __init__(self, args):
				pass
    def __call__(self, image: torch.Tensor
        ) -> Output Type:

@torch.jit.script
class PostProcess:
    """Process output values of detection models."""
    def __init__(self, args):
				pass
    def __call__(self, args) -> Output Type:
```



### Inference with the Exported Model

#### Non-End2end model

```python
input_data_list = [np.asarray(Image.open(img))]

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
input_data_list = [np.asarray(Image.open(img))]

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
