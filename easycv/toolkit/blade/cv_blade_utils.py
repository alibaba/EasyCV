# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import ctypes
import itertools
import logging
import os
import time
import timeit
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch_blade
import torch_blade.tensorrt
import torchvision
from torch_blade import optimize

os.environ['DISC_ENABLE_STITCH'] = os.environ.get('DISC_ENABLE_STITCH', 'true')
os.environ['DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE'] = os.environ.get(
    'DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE', 'true')

_cudart = ctypes.CDLL('libcudart.so')


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
        logging.info(
            'Info: your torch version is %s, your torch cuda version is %s' %
            (torch_version, torch_cuda))

    try:
        import torch_blade.tensorrt
    except:
        env_flag = False
        logging.error(
            'Import torch_blade.tensorrt failed, Install torch_blade.tensorrt and export  xx/tensorrt.so to your python ENV'
        )

    logging.info(
        'Welcome to use torch_blade, with torch %s, cuda %s, blade %s' %
        (torch_version, torch_cuda, torch_blade.version.__version__))

    return env_flag


@contextmanager
def opt_trt_config(
        input_config=dict(enable_fp16=True, fp16_fallback_op_ratio=0.05)):
    from torch_blade import tensorrt
    torch_config = torch_blade.Config()

    BLADE_CONFIG_DEFAULT = dict(
        optimization_pipeline='TensorRT',
        enable_fp16=True,
        customize_op_black_list=[
            # 'aten::select', 'aten::index', 'aten::slice', 'aten::view', 'aten::upsample'
        ],
        fp16_fallback_op_ratio=0.05,
    )
    BLADE_CONFIG_KEYS = list(BLADE_CONFIG_DEFAULT.keys())

    for key in BLADE_CONFIG_DEFAULT.keys():
        setattr(torch_config, key, BLADE_CONFIG_DEFAULT[key])
        logging.info('setting blade torch_config %s to %s by default' %
                     (key, BLADE_CONFIG_DEFAULT[key]))

    for key in input_config.keys():
        if key in BLADE_CONFIG_KEYS:
            setattr(torch_config, key, input_config[key])
            logging.warning(
                'setting blade torch_config %s to %s by user config' %
                (key, input_config[key]))

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
        dummy = torch.cuda.amp.autocast(True)
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


def computeStats(backend, timings, batch_size=1, model_name='default'):
    """
    compute the statistical metric of time and speed

    Args:
        backend (str):  backend name
        timings (List): time list
        batch_size (int)： image batch
        model_name (str): tested model name
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
def benchmark(model, inp, backend, batch_size, model_name='default', num=200):
    """
    evaluate the time and speed of different models

    Args:
        model: input model
        inp: input of the model
        backend (str):  backend name
        batch_size (int)： image batch
        model_name (str): tested model name
        num: test forward times
    """

    torch.cuda.synchronize()
    timings = []
    for i in range(num):
        start_time = timeit.default_timer()
        model(*inp)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        meas_time = end_time - start_time
        timings.append(meas_time)

    return computeStats(backend, timings, batch_size, model_name)


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


def blade_optimize(speed_test_model,
                   model,
                   inputs,
                   blade_config=dict(
                       enable_fp16=True, fp16_fallback_op_ratio=0.05),
                   backend='TensorRT',
                   batch=1,
                   warm_up_time=10,
                   compute_cost=True,
                   use_profile=False,
                   check_result=False,
                   static_opt=True):

    if not static_opt:
        logging.info(
            'PAI-Blade use dynamic optimize for input model, export model is build for dynamic shape input'
        )
        with opt_trt_config(blade_config):
            opt_model = optimize(
                model,
                allow_tracing=True,
                model_inputs=tuple(inputs),
            )
    else:
        logging.info(
            'PAI-Blade use static optimize for input model, export model must be used as static shape input'
        )
        from torch_blade.optimization import _static_optimize
        with opt_trt_config(blade_config):
            opt_model = _static_optimize(
                model,
                allow_tracing=True,
                model_inputs=tuple(inputs),
            )

    if compute_cost:
        results = []
        inputs_t = inputs

        # end2end model and scripts needs different channel purmulate, encounter this problem only when we use end2end export
        if (inputs_t[0].shape[-1] == 3):
            shape_length = len(inputs_t[0].shape)
            if shape_length == 4:
                inputs_t = inputs_t[0].permute(0, 3, 1, 2)
                inputs_t = [inputs_t]

            if shape_length == 3:
                inputs_t = inputs_t[0].permute(2, 0, 1)
                inputs_t = (torch.unsqueeze(inputs_t, 0), )

        results.append(
            benchmark(speed_test_model, inputs_t, backend, batch, 'easycv'))
        results.append(
            benchmark(model, inputs, backend, batch, 'easycv script'))
        results.append(benchmark(opt_model, inputs, backend, batch, 'blade'))

        logging.info('Model Summary:')
        summary = pd.DataFrame(results)
        logging.warning(summary.to_markdown())

    if use_profile:
        torch.cuda.empty_cache()
        # warm-up
        for k in range(warm_up_time):
            test_result = opt_model(*inputs)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        cu_prof_start()
        for k in range(warm_up_time):
            test_result = opt_model(*inputs)
            torch.cuda.synchronize()
        cu_prof_stop()
        import torch.autograd.profiler as profiler
        with profiler.profile(use_cuda=True) as prof:
            for k in range(warm_up_time):
                test_result = opt_model(*inputs)
                torch.cuda.synchronize()

        with profiler.profile(use_cuda=True) as prof:
            for k in range(warm_up_time):
                test_result = opt_model(*inputs)
                torch.cuda.synchronize()

        prof_str = prof.key_averages().table(sort_by='cuda_time_total')
        print(f'{prof_str}')

    if check_result:
        output = model(*inputs)
        test_result = opt_model(*inputs)
        check_results(output, test_result)

    return opt_model
