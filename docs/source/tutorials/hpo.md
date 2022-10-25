# HPO tutorial

Auto hyperparameter optimization (HPO), or auto tuning, is one of the key features of NNI. The tutorial gives examples of EasyCV using HPO.

## Create environment

local:

Create DSW/ECS.

dlc:

Create NAS disks, NAS datasets, and DSW/ECS (ps: Note that the three parts are created in the same region).

Mount NAS disks on DSW/ECS (ps: The address where the NAS is mounted can be the same as the mount path /mnt/data where the NAS data set is created to avoid errors).

## Installation

```shell
hpo_tools:
pip install https://automl-nni.oss-cn-beijing.aliyuncs.com/nni/hpo_tools/hpo_tools-0.1.1-py3-none-any.whl

dlc_tools(options):
wget https://automl-nni.oss-cn-beijing.aliyuncs.com/nni/hpo_tools/scripts/install_dlc.sh
source install_dlc.sh /mnt/data https://dlc-tools.oss-cn-zhangjiakou.aliyuncs.com/release/linux/dlc?spm=a2c4g.11186623.0.0.1b9b4a35er7EfB
# test
cd /mnt/data/software
dlc --help
```

## RUN
Take easycv/toolkit/hpo/search/det/ as an example

```shell
cd  EasyCV/easycv/toolkit/hpo/det/

local:
nnictl create --config config_local.yml --port=8780

dlc:
nnictl create --config config_dlc.yml --port=8780


## STOP
nnictl stop
```

For more nnictl usage, see https://nni.readthedocs.io/en/v2.1/Tutorial/QuickStart.html.

## *.yml file parameter meaning (using config_dlc.yml as an example)
```shell
experimentWorkingDirectory: ./expdir
searchSpaceFile: search_space.json
trialCommand: python3 ../common/run.py --config=./config_dlc.ini
trialConcurrency: 1
maxTrialNumber: 4
debug: true
logLevel: debug
trainingService:
  platform: local
tuner:
  name: TPE
  classArgs:
    optimize_mode: maximize
assessor:
   codeDirectory: hpo_tools的安装根目录/hpo_tools/core/assessor
   className: dlc_assessor.DLCAssessor
   classArgs:
      optimize_mode: maximize
      start_step: 2
```
<details>
<summary>Arguments</summary>

- `ExperimentWorkingDirectory`: the save directory
- `searchSpaceFile`: the search space
- `trialCommand`: startup scripts run.py(--config specified config path)
- `trainingService.platform`: the training platform
- `tuner`: the tuner algorithm
- `assessor`: the assessor algorithm
- `classArgs`: the algorithm parameters

</details>

The search space can reference: https://nni.readthedocs.io/en/v2.2/Tutorial/SearchSpaceSpec.html.

## *.ini file parameter meaning (using config_dlc.ini as an example)
```shell
[cmd_config]
cmd1="dlc config --access_id xxx --access_key xxx --endpoint 'pai-dlc.cn-shanghai.aliyuncs.com' --region cn-shanghai"
cmd2="dlc submit pytorch --name=test_nni_${exp_id}_${trial_id} \
        --workers=1   \
        --worker_cpu=12 \
        --worker_gpu=1 \
        --worker_memory=10Gi \
        --worker_spec='ecs.gn6v-c10g1.20xlarge' \
        --data_sources='d-domlyt834bngpr68iu' \
        --worker_image=registry-vpc.cn-shanghai.aliyuncs.com/mybigpai/nni:0.0.3  \
        --command='cd ../../../../../ && pip install mmcv-full && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
        && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29400 tools/train.py easycv/toolkit/hpo/search/det/fcos_r50_torch_1x_coco.py --work_dir easycv/toolkit/hpo/search/det/model/model_${exp_id}_${trial_id} --launcher pytorch   --seed 42 --deterministic --user_config_params --data_root /root/data/coco/ --data.imgs_per_gpu ${batch_size} --optimizer.lr ${lr} ' \
        --workspace_id='255705' "

[metric_config]
metric_filepath=easycv/toolkit/hpo/search/det/model/model_${exp_id}_${trial_id}/tf_logs
val/DetectionBoxes_Precision/mAP=100
```
<details>
<summary>Arguments</summary>

cmd1 specifies the area for the dlc, and cmd2 is the dlc startup command.

[cmd_config]
- `access_id and access_key`: the ak information
- `endpoint`: the port
- `region`: the region
- `name`: the experiment name
- `workers`: the number of machines
- `worker_cpu`: the number of cpus
- `worker_gpu`: the number of gpus
- `worker_memory`: the number of memory required
- `worker_spec`: the model of the machine
- `data_sources`: mapping mounts the nas, and the dlc is started using the data_sources code
- `worker_image`: the image to use
- `workspace_id`: the workspace
- `command`: the command to start the easycv experiment
- `user_config_param`: parameter is selected from searchspace.json

[metric_config]
- `metric_filepath`: tf_logs directory saved for the experiment and used to obtain the parameters of the hpo evaluation

For example, the above example uses the detected map as the evaluation parameter, with a maximum value of 100.

</details>

For details about the dlc command parameters, see https://yuque.antfin-inc.com/pai-user/manual/eo7doa.

Tuning method can be reference NNI way of use: https://nni.readthedocs.io/en/v2.1/Overview.html.
