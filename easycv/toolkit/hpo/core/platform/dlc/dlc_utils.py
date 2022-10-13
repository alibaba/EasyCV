import json
import logging
import os
import re
import time

from hpo_tools.core.utils.json_utils import get_value, set_value


def submit_job(cmd):
    """ submit result like this:
   {
    "CodeSource": {
       "Branch": "",
       "CodeSourceId": "",
       "Commit": ""
    },
    "DataSources": [
       {
          "DataSourceId": "d-sgqhx2gsfq27taohn4"
       }
    ],
    "DisplayName": "test_nni_v1ma05p9_NviMQ",
    "JobSpecs": [
       {
          "EcsSpec": "",
          "Image": "registry-vpc.cn-shanghai.aliyuncs.com/mybigpai/nni:0.0.3",
          "PodCount": 1,
          "ResourceConfig": {
             "CPU": "12",
             "GPU": "1",
             "GPUType": "",
             "Memory": "10Gi",
             "SharedMemory": ""
          },
          "Type": "Worker"
       }
    ],
    "JobType": "PyTorchJob",
    "Priority": 1,
    "ResourceId": "rg19d2oleg252kke",
    "ThirdpartyLibDir": "",
    "UserCommand": "python /mnt/data/examples/search/dlc_mnist/mnist.py --data_dir=/mnt/data/examples/search/data --save_model=/mnt/data/exmaples/search/model/model_v1ma05p9_NviMQ --batch_size=32 --lr=0.0001 --metric_filepath=/mnt/data/examples/search/metric/metric_v1ma05p9_NviMQ ",
    "WorkspaceId": "259315"
 }
 +------------------+--------------------------------------+
 |      JobId       |              RequestId               |
 +------------------+--------------------------------------+
 | dlc1xjcumbfm7aai | 71CB9CAA-87F6-5F8E-A38A-8D0E32F2559F |
 +------------------+--------------------------------------+
    """
    result = os.popen(cmd)
    context = result.read()
    result.close()

    print('submit result:', context)

    pattern1 = '\sdlc.*? '
    x = re.findall(pattern1, context)
    job_id = x[0].strip()

    print('job_id:', job_id)
    return job_id


def get_status(job_id):
    """dlc get job job_id result like this:
    {
   "ClusterId": "asi_cn-shanghai_pai_g01",
   "CodeSource": {
      "CodeSourceId": "",
      "MountPath": ""
   },
   "DataSources": [
      {
         "DataSourceId": "d-sgqhx2gsfq27taohn4",
         "MountPath": ""
      }
   ],
   "DisplayName": "test_nni_v1ma05p9_NviMQ",
   "Duration": 186,
   "ElasticSpec": {
      "AIMasterType": "",
      "EnableElasticTraining": false,
      "MaxParallelism": 0,
      "MinParallelism": 0
   },
   "EnabledDebugger": false,
   "GmtCreateTime": "2022-09-05T10:03:38Z",
   "GmtFinishTime": "2022-09-05T10:06:44Z",
   "GmtRunningTime": "2022-09-05T10:03:51Z",
   "GmtSubmittedTime": "2022-09-05T10:03:50Z",
   "GmtSuccessedTime": "2022-09-05T10:06:44Z",
   "JobId": "dlc1xjcumbfm7aai",
   "JobSpecs": [
      {
         "AssignNodeSpec": {
            "EnableAssignNode": false,
            "NodeNames": ""
         },
         "EcsSpec": "",
         "Image": "registry-vpc.cn-shanghai.aliyuncs.com/mybigpai/nni:0.0.3",
         "PodCount": 1,
         "ResourceConfig": {
            "CPU": "12",
            "GPU": "1",
            "GPUType": "",
            "Memory": "10Gi",
            "SharedMemory": ""
         },
         "Type": "Worker",
         "UseSpotInstance": false
      }
   ],
   "JobType": "PyTorchJob",
   "Pods": [
      {
         "GmtCreateTime": "2022-09-05T10:03:51Z",
         "GmtFinishTime": "2022-09-05T10:06:42Z",
         "GmtStartTime": "2022-09-05T10:03:55Z",
         "Ip": "10.224.117.178",
         "PodId": "dlc1xjcumbfm7aai-master-0",
         "PodUid": "a525e269-1553-4cfa-a107-ebfcf1285ff4",
         "Status": "Succeeded",
         "Type": "master"
      }
   ],
   "ReasonCode": "JobSucceeded",
   "ReasonMessage": "PyTorchJob dlc1xjcumbfm7aai is successfully completed.",
   "RequestId": "A8C4F08A-6CB2-504C-9654-CF59B68EB73A",
   "ResourceId": "rg19d2oleg252kke",
   "Priority": 1,
   "ResourceLevel": "L0",
   "Settings": {
      "BusinessUserId": "",
      "Caller": "",
      "EnableErrorMonitoringInAIMaster": false,
      "EnableTideResource": false,
      "ErrorMonitoringArgs": "",
      "PipelineId": ""
   },
   "Status": "Succeeded",
   "ThirdpartyLibDir": "",
   "UserCommand": "python /mnt/data/examples/search/dlc_mnist/mnist.py --data_dir=/mnt/data/examples/search/data --save_model=/mnt/data/exmaples/search/model/model_v1ma05p9_NviMQ --batch_size=32 --lr=0.0001 --metric_filepath=/mnt/data/examples/search/metric/metric_v1ma05p9_NviMQ ",
   "UserId": "1157703270994901",
   "WorkspaceId": "259315",
   "WorkspaceName": "yuze_demo_workspace"
}
    """
    cmd = 'dlc get job ' + job_id
    result = os.popen(cmd)
    context = result.read()
    result.close()

    status_dict = json.loads(context)
    status = status_dict['Status']

    return status


def stop_job(job_id):
    cmd = 'dlc stop job ' + job_id + ' --force'
    result = os.popen(cmd)
    context = result.read()
    result.close()

    print('stop job result:', context)


def get_job(job_id):
    while True:
        try:
            status = get_status(job_id).upper()
            print('job_id:', job_id, 'status:' + status)
            if status in ['COMPLETED', 'SUCCEEDED', 'FAILED', 'STOPPED']:
                break
            # to avoid user flow control
            time.sleep(60)
        except Exception as e:
            logging.exception('dlc get status error: \n')

    logging.info('exit job_id %s update status', job_id)
    return status


def kill_job(trial_id):
    while True:
        try:
            job_id = get_value(trial_id, trial_id=trial_id)
            print('job_id:', job_id)
            if job_id:
                stop_job(job_id)
            break
            # to avoid user flow control
            time.sleep(60)
        except Exception as e:
            logging.exception('dlc stop error: \n')


def run_multi_command(cmd_config, trial_id=None):
    # parse command
    for k, cmd in cmd_config.items():
        cmd = cmd.strip().strip('"').strip("'")
        print(cmd)
        if cmd.strip().lower().startswith('dlc submit') or cmd.strip().lower(
        ).startswith('dlc create job'):
            job_id = submit_job(cmd)
            set_value(trial_id, str(job_id), trial_id=trial_id)
            status = get_job(job_id)
            if status == 'FAILED':
                exit(1)
        else:
            ret = os.system(cmd)
            print('retcode:', ret)

            if ret:
                exit(1)
