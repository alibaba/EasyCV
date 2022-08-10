# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import subprocess
import sys

easycv_root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def train(config_path, gpus=1, fp16=False, master_port=29527):

    tpath = config_path

    current_env = os.environ.copy()

    cmd = [sys.executable, '-m', 'torch.distributed.launch']

    cmd.append('--nproc_per_node={}'.format(gpus))

    cmd.append('--master_port={}'.format(master_port))

    training_script = os.path.join(easycv_root_path, 'tools/train.py')
    cmd.append(training_script)

    cmd.append('{}'.format(tpath))

    config_file_name = os.path.basename(tpath)
    work_dir = os.path.splitext(config_file_name)[0]
    cmd.append('--work_dir={}'.format(work_dir))

    cmd.append('--launcher=pytorch')

    if fp16:
        cmd.append('--fp16')

    process = subprocess.Popen(cmd, env=current_env)

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=process.returncode, cmd=cmd)


def eval(config_path, checkpoint_path, gpus=1, fp16=False, master_port=29600):

    tpath = config_path

    current_env = os.environ.copy()

    cmd = [sys.executable, '-m', 'torch.distributed.launch']

    cmd.append('--nproc_per_node={}'.format(gpus))

    cmd.append('--master_port={}'.format(master_port))

    eval_script = os.path.join(easycv_root_path, 'tools/eval.py')
    cmd.append(eval_script)

    cmd.append('{}'.format(tpath))

    cmd.append('{}'.format(checkpoint_path))

    cmd.append('--launcher=pytorch')

    cmd.append('--eval')

    if fp16:
        cmd.append('--fp16')

    process = subprocess.Popen(cmd, env=current_env)

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=process.returncode, cmd=cmd)


def export(config_path, checkpoint_path, export_path):

    tpath = config_path

    export_script = os.path.join(easycv_root_path, 'tools/export.py')
    cmd = [sys.executable, export_script]

    cmd.append('{}'.format(tpath))

    cmd.append('{}'.format(checkpoint_path))

    cmd.append('{}'.format(export_path))

    process = subprocess.Popen(cmd)

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=process.returncode, cmd=cmd)
