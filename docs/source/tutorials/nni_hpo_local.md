# NNI HPO local tutorial

Auto hyperparameter optimization (HPO), or auto tuning, is one of the key features of NNI. This tutorial shows an example of EasyCV for local using NNI HPO.

## Create environment

Create DSW/ECS.

For details about the create environment, see https://yuque.antfin.com/pai-user/manual/rwk4sh.

## Installation

```shell
hpo_tools:
pip install https://automl-nni.oss-cn-beijing.aliyuncs.com/nni/hpo_tools/hpo_tools-0.1.1-py3-none-any.whl

```

## RUN
Take easycv/toolkit/hpo/search/det/ as an example

```shell
cd  EasyCV/easycv/toolkit/hpo/det/

nnictl create --config config_local.yml --port=8780

## STOP
nnictl stop
```

For more nnictl usage, see https://nni.readthedocs.io/en/v2.1/Tutorial/QuickStart.html.

## config_local.yml file parameter meaning
```shell
experimentWorkingDirectory: ./expdir
searchSpaceFile: search_space.json
trialCommand: python3 ../common/run.py --config=./config_local.ini
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
   codeDirectory: /root/anaconda3/lib/python3.9/site-packages/hpo_tools/core/assessor
   className: dlc_assessor.DLCAssessor
   classArgs:
      optimize_mode: maximize
      start_step: 2
      moving_avg: true
      proportion: 0.6
      patience: 2
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

## config_local.ini file parameter meaning
```shell
[cmd_config]
cmd1='cd /mnt/data/EasyCV && CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29400 tools/train.py easycv/toolkit/hpo/det/fcos_r50_torch_1x_coco.py --work_dir easycv/toolkit/hpo/det/model/model_${exp_id}_${trial_id} --launcher pytorch   --seed 42 --deterministic --user_config_params --data_root /mnt/data/coco/ --data.imgs_per_gpu ${batch_size} --optimizer.lr ${lr} '

[metric_config]
metric_filepath=easycv/toolkit/hpo/det/model/model_${exp_id}_${trial_id}/tf_logs
val/DetectionBoxes_Precision/mAP=100
```
<details>
<summary>Arguments</summary>

cmd1 is a local run command.

[cmd_config]
- `user_config_param`: parameter is selected from searchspace.json

[metric_config]
- `metric_filepath`: tf_logs directory saved for the experiment and used to obtain the parameters of the hpo evaluation

For example, the above example uses the detected map as the evaluation parameter, with a maximum value of 100.

</details>

Tuning method can be reference NNI way of use: https://nni.readthedocs.io/en/v2.1/Overview.html.
