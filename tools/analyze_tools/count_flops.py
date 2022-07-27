# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import inspect
from collections import Counter

import numpy as np
import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_table
from mmcv.parallel import scatter_kwargs
from prettytable import PrettyTable

from easycv.datasets.builder import build_dataset
from easycv.datasets.loader import build_dataloader
from easycv.datasets.utils import is_dali_dataset_type
from easycv.models.builder import build_model
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab


def parse_args():
    parser = argparse.ArgumentParser(description='count model flops')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--repeat_num', default=10, type=int, help='repeat number')
    args = parser.parse_args()
    return args


def flatten_inputs(model, inputs):
    full_args_spec = inspect.getfullargspec(model.forward)
    args = [] if not full_args_spec.args else full_args_spec.args
    args.pop(0) if (args and args[0] in ['self', 'cls']) else args

    default_values = [] if not full_args_spec.defaults else full_args_spec.defaults
    args_has_default = args[len(args) - len(default_values):]
    args_with_default = dict(zip(
        args_has_default, default_values)) if len(args_has_default) else {}

    flat_inputs = []
    for arg_i in args:
        if inputs.get(arg_i, None) is not None:
            flat_inputs.append(inputs.get(arg_i))
        else:
            flat_inputs.append(args_with_default.get(arg_i))

    return tuple(flat_inputs)


def count_flop():
    args = parse_args()

    device = 'cuda'
    cfg = mmcv_config_fromfile(args.config)

    # dynamic adapt mmdet models
    dynamic_adapt_for_mmlab(cfg)

    model = build_model(cfg.model)
    model.to(device)
    model.eval()

    if cfg.data.get('val', None) is not None:
        cfg.data.val.pop('imgs_per_gpu', None)  # pop useless params
        data_cfg = cfg.data.val
    else:
        data_cfg = cfg.data.train

    if is_dali_dataset_type(data_cfg['type']):
        data_cfg.distributed = False
        data_cfg.batch_size = 1
        data_cfg.workers_per_gpu = 1
        dataset = build_dataset(data_cfg)
        data_loader = dataset.get_dataloader()
    else:
        dataset = build_dataset(data_cfg)
        data_loader = build_dataloader(
            dataset, imgs_per_gpu=1, workers_per_gpu=0)

    handlers = {}  # mapping from operator names to handles.
    counts = Counter()
    gflop_unit = 1e9
    total_flops = []
    for idx, data in zip(tqdm.trange(args.repeat_num), data_loader):
        # use scatter_kwargs to unpack DataContainer data for raw torch.nn.module
        _, kwargs = scatter_kwargs(None, data, [0])
        kwargs[0].update({'mode': 'test'})
        inputs = flatten_inputs(model, kwargs[0])

        # Provides access to per-submodule model flop count obtained by
        # tracing a model with pytorch's jit tracing functionality.
        # So models that donot support jit tracing may fail.
        flops = FlopCountAnalysis(model, inputs)
        flops.set_op_handle(**handlers)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(
                False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    print('Flops from only one sample is:\n' + flop_count_table(flops))
    ops_show = PrettyTable()
    ops_show.field_names = ['operator type', 'Gflops']
    for k, v in counts.items():
        ops_show.add_row([k, round(v / (idx + 1) / gflop_unit, 3)])
    print('Average Gflops of each operator type is:')
    print(ops_show)
    print('Total flops: {:.1f}G ± {:.1f}G'.format(
        np.mean(total_flops) / gflop_unit,
        np.std(total_flops) / gflop_unit))


if __name__ == '__main__':
    count_flop()
