# TorchAccelerator tutorial

TorchAccelerator is a distributed training acceleration framework that transfers eager execution to graph-based intermediate representation on Pytorch. TorchAccelerator accelerates model training on Pytorch by means of compilation optimization and manual operator optimization.

## Preparation

Currently we only provide docker run.

### Docker

#### Prerequisites

- Driver Version: 470.82.01+
- CUDA Version: 11.3+

**Create Container**

image url: `registry.cn-hangzhou.aliyuncs.com/pai-dlc/pytorch-training:cuda11.3.1-cudnn8-devel-ubuntu20.04-py38-0625`

```shell
$ nvidia-docker run -it --name $YOUR_NAME --gpus all -v ${YOUR_ROOT_DIR}:/workspace registry.cn-hangzhou.aliyuncs.com/pai-dlc/pytorch-training:cuda11.3.1-cudnn8-devel-ubuntu20.04-py38-0625 bash
```

**Prepare EasyCV**

Refer to: [quick_start.md](https://github.com/alibaba/EasyCV/blob/master/docs/source/quick_start.md)

## RUN

**The first few steps to run initialization will be very slow, please be patient.**

### Single Gpu

```shell
$ USE_TORCHACC=1 python tools/train.py configs/classification/imagenet/swint/imagenet_swin_tiny_patch4_window7_224_jpg_torchacc.py --work_dir ./work_dirs  --fp16
```

### Multi Gpus

```shell
$ USE_TORCHACC=1 xlarun --nproc_per_node=${NUM_GPUS} --master_port=29500 tools/train.py configs/classification/imagenet/swint/imagenet_swin_tiny_patch4_window7_224_jpg_torchacc.py --work_dir ./work_dirs  --fp16
```

## Benchmark

### Single Gpu

Device: Tesla V100

The throughput is as follows:

|      | Raw    | Torchacc | Speedup    |                                 |
| ---- | ------ | -------- | ---------- | ------------------------------- |
| Swin | 319.68 | 582.94   | **82.35%** | batch_size=160 (per gpu) / fp16 |

### Multi Gpus

Device: Tesla V100

The throughput of 8 gpus is as follows:

|      | Raw  | Torchacc | Speedup |                                 |
| ---- | ---- | -------- | ------- | ------------------------------- |
| Swin | 2250 | 3462.7   | **54%** | batch_size=160 (per gpu) / fp16 |
