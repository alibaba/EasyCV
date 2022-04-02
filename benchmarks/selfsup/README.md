# Quick Start

## Linear Evaluation

### Method 1: Extract features（recommend）

This method is highly efficient.

It first extracts data features with pretrained models,  then performs linear eval on the features.

This means there is no data augments and all feature layers are frozen.

**Step1: Extract data features with pretrained model.**

```shell
sh benchmarks/tools/dist_extract.sh \
        ${EXTRACT_DATASET_CONFIG} \
        ${NUM_GPUS} \
        ${FEATURE_DIR} \
        --checkpoint ${CHECKPOINT}
```

<details>
<summary>Arguments</summary>

- `EXTRACT_DATASET_CONFIG`:the config path of extract data features, refer to [extract_dataset_configs](benchmarks/extract_dataset_configs).
-  `NUM_GPUS`: number of gpus
- `FEATURE_DIR`:your path to save output features
- `CHECKPOINT`: the export checkpoint file of a selfsup model named as epoch\_\*\_export.pt.

</details>

**Examples:**

Export model please reference to [ssl.md](../docs/source/tutorials/ssl.md), or you can test with our default export model [swav_restnet50](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/swav_r50/epoch_200_export.pt -O ./pretrained_models/swav_rn50/epoch_200_export.pt).

```shell
sh benchmarks/tools/dist_extract.sh \
        benchmarks/extract_dataset_configs/imagenet.py \
        8 \
        ./linear_eval/imagenet_features \
        --checkpoint ./pretrained_models/swav_rn50/epoch_200_export.pt
```

**Step2: Do linear evaluation with features.**

```shell
sh tools/dist_train.sh \
        ${FEATURE_LINEAR_EVAL_CONFIG_PATH} \
        ${NUM_GPUS} \
        --work_dir ${WORK_DIR}
```

<details>
<summary>Arguments</summary>

- `FEATURE_LINEAR_EVAL_CONFIG_PATH`:the config path of linear eval with features.

  Reference to `benchmarks/selfsup/classification/imagenet`,edit feature path to your local or oss path.

-  `NUM_GPUS`: number of gpus

- `WORK_DIR`:your path to save models and logs

</details>

**Examples:**

Edit `data_root` in `${FEATURE_LINEAR_EVAL_CONFIG_PATH}` to your own `${FEATURE_DIR}` path.

```shell
sh tools/dist_train.sh \
        benchmarks/selfsup/imagenet/swav_r50_feature.py \
        8 \
        --work_dir ./linear_eval
```

### Method 2: Finetune with fc layer

```shell
sh tools/dist_train.sh \
        ${LINEAR_EVAL_CONFIG_PATH} \
        ${NUM_GPUS} \
        --work_dir ${WORK_DIR}
        --load_from ${LOAD_FROM}
```

<details>
<summary>Arguments</summary>

- `LINEAR_EVAL_CONFIG_PATH`: the config path of linear eval
-  `NUM_GPUS`: number of gpus
- `WORK_DIR`: your path to save models and logs
- `LOAD_FROM`: the pretrained checkpoint file of a selfsup model named as epoch\_\*.pth.

</details>

**Examples:**

Edit the `data_root` in the `${LINEAR_EVAL_CONFIG_PATH}` to your own data path.

`${LOAD_FROM}` can use your own pretrained model, or you can test with our default pretrained model above.

```shell
sh tools/dist_train.sh \
        benchmarks/selfsup/classification/imagenet/resnet50_8xb32_100e_finetune.py \
        8 \
        --work_dir ./linear_eval
        --load_from ./pretrained_models/swav_rn50/epoch_200.pth
```
