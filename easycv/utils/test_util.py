# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contains functions which are convenient for unit testing."""
import logging
import os
import shutil
import socket
import subprocess
import timeit
import uuid
from multiprocessing import Process

import numpy as np
import torch

TEST_DIR = '/tmp/ev_pytorch_test'


def get_tmp_dir():
    if os.environ.get('TEST_DIR', '') != '':
        global TEST_DIR
        TEST_DIR = os.environ['TEST_DIR']
    dir_name = os.path.join(TEST_DIR, uuid.uuid4().hex)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    return dir_name


def clear_all_tmp_dirs():
    shutil.rmtree(TEST_DIR)


def replace_data_for_test(cfg):
    """
  replace real data with test data

  Args:
    cfg: Config object
  """
    pass


# function dectorator to run function in subprocess
# if a function will start a tf session. Because tensorflow
# gpu memory will not be cleared until the process exit
def RunAsSubprocess(f):

    def wrapped_f(*args, **kw):
        p = Process(target=f, args=args, kwargs=kw)
        p.start()
        p.join(timeout=600)
        assert p.exitcode == 0, 'subprocess run failed: %s' % f.__name__

    return wrapped_f


def clean_up(test_dir):
    if test_dir is not None:
        shutil.rmtree(test_dir)


def run_in_subprocess(cmd):
    try:
        with subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT) as return_info:
            while True:
                next_line = return_info.stdout.readline()
                return_line = next_line.decode('utf-8', 'ignore').strip()
                if return_line == '' and return_info.poll() != None:
                    break
                if return_line != '':
                    logging.info(return_line)

            return_code = return_info.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, return_info)
    except Exception as e:
        raise e


def dist_exec_wrapper(cmd,
                      nproc_per_node,
                      node_rank=0,
                      nnodes=1,
                      port='29527',
                      addr='127.0.0.1',
                      python_path=None):
    """
    donot forget init dist in your function or script of cmd
    ```python
    from mmcv.runner import init_dist
    init_dist(launcher='pytorch')
    ```
    """

    dist_world_size = nproc_per_node * nnodes
    cur_env = os.environ.copy()
    cur_env['WORLD_SIZE'] = str(dist_world_size)
    cur_env['MASTER_ADDR'] = addr

    if is_port_used(port):
        port = str(get_random_port())
        logging.warning('Given port is used, change to port %s' % port)

    cur_env['MASTER_PORT'] = port
    processes = []

    for local_rank in range(0, nproc_per_node):
        # rank of each process
        dist_rank = nproc_per_node * node_rank + local_rank
        cur_env['RANK'] = str(dist_rank)
        cur_env['LOCAL_RANK'] = str(local_rank)
        if python_path:
            cur_env['PYTHONPATH'] = ':'.join(
                (cur_env['PYTHONPATH'], python_path))

        process = subprocess.Popen(cmd, env=cur_env, shell=True)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=cmd)


def is_port_used(port, host='127.0.0.1'):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, int(port)))
        return True
    except:
        return False
    finally:
        s.close()


def get_random_port():
    while True:
        port = np.random.randint(low=5000, high=10000)
        if is_port_used(port):
            continue
        else:
            break

    logging.info('Random port: %s' % port)

    return port


def pseudo_dist_init():
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(get_random_port())
    torch.cuda.set_device(0)
    from torch import distributed as dist
    dist.init_process_group(backend='nccl')


def computeStats(backend, timings, batch_size=1, model_name='default'):
    """
    compute the statistical metric of time and speed
    """
    times = np.array(timings)
    steps = len(times)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    msg = ('\n%s =================================\n'
           'batch size=%d, num iterations=%d\n'
           '  Median FPS: %.1f, mean: %.1f\n'
           '  Median latency: %.6f, mean: %.6f, 99th_p: %.6f, std_dev: %.6f\n'
           ) % (
               backend,
               batch_size,
               steps,
               speed_med,
               speed_mean,
               time_med,
               time_mean,
               time_99th,
               time_std,
           )

    meas = {
        'Name': model_name,
        'Backend': backend,
        'Median(FPS)': speed_med,
        'Mean(FPS)': speed_mean,
        'Median(ms)': time_med,
        'Mean(ms)': time_mean,
        '99th_p': time_99th,
        'std_dev': time_std,
    }

    return meas


@torch.no_grad()
def benchmark(predictor,
              input_data_list,
              backend='BACKEND',
              batch_size=1,
              model_name='default',
              num=200):
    """
    evaluate the time and speed of different models
    """

    timings = []
    for i in range(num):
        start_time = timeit.default_timer()
        output = predictor.predict(input_data_list)[0]
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        meas_time = end_time - start_time
        timings.append(meas_time)

    return computeStats(backend, timings, batch_size, model_name)
