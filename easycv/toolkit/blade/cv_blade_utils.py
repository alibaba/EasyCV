import argparse
import itertools
import os
import time
import timeit
import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torchvision

os.environ['DISC_ENABLE_STITCH'] = os.environ.get('DISC_ENABLE_STITCH', 'true')
os.environ['DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE'] = os.environ.get('DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE', 'true')

try:
    import torch_blade
    import torch_blade.tensorrt
    from torch_blade import optimize
    import ctypes
    _cudart = ctypes.CDLL('libcudart.so')
except:
    pass


def blade_env_assert():
    env_flag = True

    try:
        import torch
        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
    except:
        torch_version = 'failed'
        torch_cuda = 'failed'
        env_flag = False
        logging.error(
            'import torch and torch cuda failed, please install pytorch with cuda correctly'
        )

    try:
        import torch_blade
    except:
        env_flag = False
        logging.error(
            'Import torch_blade failed, please reference to https://help.aliyun.com/document_detail/205134.html'
        )
        logging.info('Info: your torch version is %s, your torch cuda version is %s' %
              (torch_version, torch_cuda))

    try:
        import torch_blade.tensorrt
    except:
        env_flag = False
        logging.error(
            'Import torch_blade.tensorrt failed, Install torch_blade.tensorrt and export  xx/tensorrt.so to your python ENV'
        )

    logging.info('Welcome to use torch_blade, with torch %s, cuda %s, blade %s' %
          (torch_version, torch_cuda, torch_blade.version.__version__))

    return env_flag


@contextmanager
def opt_trt_config(enable_fp16=True):
    from torch_blade import tensorrt
    torch_config = torch_blade.Config()
    torch_config.optimization_pipeline = 'TensorRT'
    torch_config.enable_fp16 = enable_fp16
    torch_config.customize_op_black_list = [
        'aten::select', 'aten::index', 'aten::slice', 'aten::view'
    ]
    torch_config.fp16_fallback_op_ratio = 0.3
    try:
        with torch_config:
            yield
    finally:
        pass


def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)


@contextmanager
def opt_blade_mixprec():
    try:
        dummy = torch.classes.torch_blade.MixPrecision(True)
        yield
    finally:
        pass


@contextmanager
def opt_disc_config(enable_fp16=True):
    torch_config = torch_blade.config.Config()
    torch_config.enable_fp16 = enable_fp16
    try:
        with torch_config:
            yield
    finally:
        pass


results = []


def printStats(backend, timings, batch_size=1, model_name='default'):
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
    # print(msg)
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
    results.append(meas)


@torch.no_grad()
def benchmark(model, inp, backend, batch_size, model_name='default'):
    for i in range(100):
        model(*inp)
    torch.cuda.synchronize()
    timings = []
    for i in range(200):
        start_time = timeit.default_timer()
        model(*inp)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        meas_time = end_time - start_time
        timings.append(meas_time)

    printStats(backend, timings, batch_size, model_name)


def collect_tensors(data):
    if isinstance(data, torch.Tensor):
        return [data]
    elif isinstance(data, list):
        return list(itertools.chain(*[collect_tensors(d) for d in data]))
    elif isinstance(data, dict):
        sorted_pairs = sorted(data.items(), key=lambda x: x[0])
        sorted_list = [v for k, v in sorted_pairs]
        return collect_tensors(sorted_list)
    elif isinstance(data, tuple):
        return collect_tensors(list(data))
    else:
        return []


def check_results(results0, results1):
    from torch_blade.testing.common_utils import assert_almost_equal

    results0 = collect_tensors(results0)
    results1 = collect_tensors(results1)

    try:
        assert_almost_equal(results0, results1, rtol=1e-3, atol=1e-3)
        logging.info('Accuraccy check passed')
    except Exception as err:
        logging.error(err)


def blade_yolox_optimize(script_model,
                         model,
                         inputs,
                         fp16=True,
                         backend='TensorRT',
                         batch=1):
    with opt_trt_config(fp16):
        opt_model = optimize(
            model,
            allow_tracing=True,
            model_inputs=tuple(inputs),
        )
    benchmark(script_model, inputs, backend, batch, 'easycv')
    benchmark(model, inputs, backend, batch, 'easycv script')
    benchmark(opt_model, inputs, backend, batch, 'blade')
    logging.info('Model Summary:')
    summary = pd.DataFrame(results)
    # print(summary.to_markdown())
    logging.info(summary.to_markdown())

    # x, y, z = inputs
    # inputs = (x.to(torch.int32), y.to(torch.int32), z.to(torch.int32))
    output = model(*inputs)
    cu_prof_start()
    if fp16:
        with opt_blade_mixprec():
            test_result = model(*inputs)
    else:
        test_result = opt_model(*inputs)
    cu_prof_stop()
    check_results(output, test_result)
    return opt_model


if __name__ == '__main__':

    print('blade test')
    blade_env_assert()
