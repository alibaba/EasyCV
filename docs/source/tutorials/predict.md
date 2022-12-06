# 批量推理
EasyCV提供了工具支持本地大规模图片推理能力，该工具支持读取本地图片、图片http链接、MaxCompute表数据，使用EasyCV提供的各类Predictor进行预测。

## 依赖安装
安装easy_predict, easy_predict把图片预测过程中的数据读取/下载、图片解码、模型推理各个部分抽象成了独立的处理单元，每个处理单元支持多线程并发执行，能够大大加速任务整体的吞吐量。
```
pip install https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/release/easy_predict-0.4.2-py2.py3-none-any.whl
```


## 数据格式
### 输入文件列表
当输入为一个文件时，文件每行可以是一个本地图片路径，也可以是图片url地址

本地文件路径
```shell
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
predict/test_data/000000289059.jpg
```

图片url
```shell
https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/ant%2Bhill_14_33.jpg
https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/ant%2Bhill_14_33.jpg
https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/ant%2Bhill_14_33.jpg
https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/ant%2Bhill_14_33.jpg
https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/ant%2Bhill_14_33.jpg
https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/ant%2Bhill_14_33.jpg
```


### 输入MaxCompute Table
输入表可以是一列或者多列， 其中一列需要是图像的url或者图像文件的二进制数据经过base64编码后的字符串(image_base64)

输入表schema示例如下：
url数据
```shell
+------------------------------------------------------------------------------------+
| Field           | Type       | Label | Comment                                     |
+------------------------------------------------------------------------------------+
| id              | string     |       |                                             |
| url             | string     |       |                                             |
+------------------------------------------------------------------------------------+
```

base64数据
输入表可以是一列或者多列， 其中一列需要是图像的url或者图像编码后的二进制数据经过base64编码的数据，type为str
schema示例如下
```shell
+------------------------------------------------------------------------------------+
| Field           | Type       | Label | Comment                                     |
+------------------------------------------------------------------------------------+
| id              | string     |       |                                             |
| base64          | string     |       |                                             |
+------------------------------------------------------------------------------------+
```


## 运行

### 读取本地文件

单卡运行
```shell
PYTHONPATH=. python tools/predict.py  \
    --input_file predict/test.list \
    --output_file predict/output.txt \
    --model_type YoloXPredictor \
    --model_path predict/test_data/yolox/epoch_300.pt
```

<details>
<summary>参数说明</summary>

- `input_file`: 输入文件路径

- `output_file`: 输出文件路径

- `model_type`: 模型类型， 对应easycv/predictors/下的不同Predictor类名， 例如YoloXPredictor

- `model_path`: 模型文件路径/模型目录
</details>

多机多卡运行

这里多机多卡启动方式复用pytorch DDP方式， 需要在GPU环境下使用
```shell
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=2 --master_port=29527 \
                    tools/predict.py \
                    --input_file predict/test.list \
                    --output_file predict/output.txt \
                    --model_type YoloXPredictor \
                    --model_path  predict/test_data/yolox/epoch_300.pt\
                    --launcher pytorch
```
<details>
<summary>参数说明</summary>

- `nproc_per_node`:  每个节点的gpu数

- `master_port`: master节点端口

- `master_addr`: master IP

- `input_file`: 输入文件路径

- `output_file`: 输出文件路径

- `model_type`: 模型类型， 对应easycv/predictors/下的不同Predictor类名， 例如YoloXPredictor

- `model_path`: 模型文件路径/模型目录

</details>


### 读取MaxComputeTable

单卡示例
```shell
#创建输出表分区
odpscmd -e "alter table 表名 add partition (ds=分区名);"

PYTHONPATH=. python tools/predict.py \
   --model_type YoloXPredictor \
   --model_path predict/test_data/yolox/epoch_300.pt \
   --input_table odps://项目名/tables/表名/ds=分区信息 \
   --output_table  odps://项目名/tables/表名/ds=分区信息\
   --image_col url\
   --image_type url\
   --reserved_columns id\
   --result_column result \
   --odps_config /path/to/odps.config
```

<details>
- `model_type`: 模型类型， 对应easycv/predictors/下的不同Predictor类名， 例如YoloXPredictor

- `model_path`: 模型文件路径/模型目录

- `input_table`: 输入表

- `output_table`: 输出表

- `image_col`: 图片数据所在列

- `image_type`: 图片类型， url or base64
- `reserved_columns`: 输入表保留列名，英文逗号分割
- `result_column`: 结果列名
- `odps_config`: MaxCompute配置文件


</details>

多卡示例

这里多机多卡启动方式复用pytorch DDP方式， 需要在GPU环境下使用
```shell
#创建输出表分区
odpscmd -e "alter table 表名 add partition (ds=分区名);"

PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=3 --master_port=29527 \
    tools/predict.py  \
    --model_type YoloXPredictor \
    --model_path predict/test_data/yolox/epoch_300.pt \
   --input_table odps://项目名/tables/表名/ds=分区名 \
   --output_table  odps://项目名/tables/表名/ds=分区名\
    --image_col url\
    --image_type url\
    --reserved_columns id\
    --result_column result \
    --odps_config  /path/to/odps.config \
    --launcher pytorch
```
