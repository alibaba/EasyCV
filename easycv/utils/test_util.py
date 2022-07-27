# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contains functions which are convenient for unit testing."""
import copy
import logging
import os
import pickle
import shutil
import socket
import subprocess
import sys
import tempfile
import timeit
import unittest
import uuid
from multiprocessing import Process

import numpy as np
import torch

from easycv.file import io

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


_DIST_SCRIPT_TEMPLATE = """
import ast
import argparse
import pickle
import torch
from torch import distributed as dist
from easycv.utils.dist_utils import is_master
import {}

parser = argparse.ArgumentParser()
parser.add_argument('--save_all_ranks', type=ast.literal_eval, help='save all ranks results')
parser.add_argument('--save_file', type=str, help='save file')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


def main():
    results = {}.{}({})  # module.func(params)

    if args.save_all_ranks:
        save_file = args.save_file + str(dist.get_rank())
        with open(save_file, 'wb') as f:
            pickle.dump(results, f)
    else:
        if is_master():
            with open(args.save_file, 'wb') as f:
                pickle.dump(results, f)


if __name__ == '__main__':
    main()
"""


class DistributedTestCase(unittest.TestCase):
    """Distributed TestCase for test function with distributed mode.
    Examples:
        import torch
        from mmcv.runner import init_dist
        from torch import distributed as dist

        def _test_func(*args, **kwargs):
            init_dist(launcher='pytorch')
            rank = dist.get_rank()
            if rank == 0:
                value = torch.tensor(1.0).cuda()
            else:
                value = torch.tensor(2.0).cuda()
            dist.all_reduce(value)
            return value.cpu().numpy()

        class DistTest(DistributedTestCase):
            def test_function_dist(self):
                args = ()  # args should be python builtin type
                kwargs = {}  # kwargs should be python builtin type
                self.start_with_torch(
                    _test_func,
                    num_gpus=2,
                    assert_callback=lambda x: self.assertEqual(x, 3.0),
                    *args,
                    **kwargs,
                )
    """

    def _start(self,
               dist_start_cmd,
               func,
               num_gpus,
               assert_callback=None,
               save_all_ranks=False,
               *args,
               **kwargs):
        script_path = func.__code__.co_filename
        script_dir, script_name = os.path.split(script_path)
        script_name = os.path.splitext(script_name)[0]
        func_name = func.__qualname__

        func_params = []
        for arg in args:
            if isinstance(arg, str):
                arg = ('\'{}\''.format(arg))
            func_params.append(str(arg))

        for k, v in kwargs.items():
            if isinstance(v, str):
                v = ('\'{}\''.format(v))
            func_params.append('{}={}'.format(k, v))

        func_params = ','.join(func_params).strip(',')

        tmp_run_file = tempfile.NamedTemporaryFile(suffix='.py').name
        tmp_res_file = tempfile.NamedTemporaryFile(suffix='.pkl').name

        with open(tmp_run_file, 'w') as f:
            print('save temporary run file to : {}'.format(tmp_run_file))
            print('save results to : {}'.format(tmp_res_file))
            run_file_content = _DIST_SCRIPT_TEMPLATE.format(
                script_name, script_name, func_name, func_params)
            f.write(run_file_content)

        tmp_res_files = []
        if save_all_ranks:
            for i in range(num_gpus):
                tmp_res_files.append(tmp_res_file + str(i))
        else:
            tmp_res_files = [tmp_res_file]
        self.addCleanup(self.clean_tmp, [tmp_run_file] + tmp_res_files)

        tmp_env = copy.deepcopy(os.environ)
        tmp_env['PYTHONPATH'] = ':'.join(
            (tmp_env.get('PYTHONPATH', ''), script_dir)).lstrip(':')
        script_params = '--save_all_ranks=%s --save_file=%s' % (save_all_ranks,
                                                                tmp_res_file)
        script_cmd = '%s %s %s' % (dist_start_cmd, tmp_run_file, script_params)
        print('script command: %s' % script_cmd)
        res = subprocess.call(script_cmd, shell=True, env=tmp_env)

        script_res = []
        for res_file in tmp_res_files:
            with open(res_file, 'rb') as f:
                script_res.append(pickle.load(f))
        if not save_all_ranks:
            script_res = script_res[0]

        if assert_callback:
            assert_callback(script_res)

        self.assertEqual(
            res,
            0,
            msg='The test function ``{}`` in ``{}`` run failed!'.format(
                func_name, script_name))

        return script_res

    def start_with_torch(self,
                         func,
                         num_gpus,
                         assert_callback=None,
                         save_all_ranks=False,
                         *args,
                         **kwargs):
        ip = socket.gethostbyname(socket.gethostname())
        dist_start_cmd = '%s -m torch.distributed.launch --nproc_per_node=%d --master_addr=\'%s\' --master_port=%s' % (
            sys.executable, num_gpus, ip, get_random_port())

        return self._start(
            dist_start_cmd=dist_start_cmd,
            func=func,
            num_gpus=num_gpus,
            assert_callback=assert_callback,
            save_all_ranks=save_all_ranks,
            *args,
            **kwargs)

    def start_with_torchacc(self,
                            func,
                            num_gpus,
                            assert_callback=None,
                            save_all_ranks=False,
                            *args,
                            **kwargs):
        ip = socket.gethostbyname(socket.gethostname())
        dist_start_cmd = 'xlarun --nproc_per_node=%d --master_addr=\'%s\' --master_port=%s' % (
            num_gpus, ip, get_random_port())

        return self._start(
            dist_start_cmd=dist_start_cmd,
            func=func,
            num_gpus=num_gpus,
            assert_callback=assert_callback,
            save_all_ranks=save_all_ranks,
            *args,
            **kwargs)

    def clean_tmp(self, tmp_file_list):
        for file in tmp_file_list:
            if io.exists(file):
                if io.isdir(file):
                    io.rmtree(file)
                else:
                    io.remove(file)
