# Quick Start

## Prerequisites
* python >= 3.6
* Pytorch  >= 1.5
* mmcv >= 1.2.0
* nvidia-dali == 0.25.0


## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n ev python=3.6 -y
    conda activate ev
    ```

2. Install PyTorch and torchvision

   The master branch works with **PyTorch 1.5.1** or higher.

    ```shell
    conda install pytorch==1.7.0 torchvision==0.8.0 -c pytorch
    ```

3. Install some python dependencies

    replace {cu_version} and {torch_version} to the version used in your environment
    ```shell
    # install mmcv
    pip install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    # for example, install mmcv-full for cuda10.1 and pytorch 1.7.0
    pip install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

    # install nvidia-dali
    pip install http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/third_party/nvidia_dali_cuda100-0.25.0-1535750-py3-none-manylinux2014_x86_64.whl

    # install common_io for MaxCompute table read (optional)
    pip install https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/third_party/common_io-0.3.0-cp36-cp36m-linux_x86_64.whl

    ```

4. Install EasyCV

    You can simply install easycv with the following command:

    ```shell
    pip install pai-easycv
    ```

    or clone the repository and then install it:
    ```shell
    git clone https://github.com/Alibaba/EasyCV.git
    cd easycv
    pip install -r requirements.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

5. Install pai_nni and blade_compressin

    When you use model quantize and prune, you need to install pai_nni and blade_compression with the following command:

    ```shell
    # install torch >= 1.8.0
    pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0

    # install mmcv >= 1.3.0 (torch version >= 1.8.0 does not support mmcv version < 1.3.0)
    pip install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

    # install onnx and pai_nni
    pip install onnx
    pip install https://pai-nni.oss-cn-zhangjiakou.aliyuncs.com/release/2.5/pai_nni-2.5-py3-none-manylinux1_x86_64.whl

    # install blade_compression
    pip install http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/third_party/blade_compression-0.0.1-py3-none-any.whl

    ```

6. If you want to use MSDeformAttn, you need to compiling CUDA operators

    ```shell
    cd thirdparty/deformable_attention/
    python setup.py build install
    # unit test (should see all checking is True)
    python test.py
    cd ../../..

    ```

### Verification

    Simple verification

    ```python
    from easycv.apis import *
    ```

    You can also verify your installation using following quick-start examples


## Examples

* [Image classification example](tutorials/cls.md)

* [Self-supervised learning example](tutorials/ssl.md)

* [object detection example](tutorials/yolox.md)
