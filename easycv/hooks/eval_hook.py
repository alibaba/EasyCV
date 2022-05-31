# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from collections import OrderedDict

import torch
from mmcv.runner import Hook
from torch.utils.data import DataLoader

from easycv.hooks.tensorboard import TensorboardLoggerHookV2
from easycv.hooks.wandb import WandbLoggerHookV2

if torch.cuda.is_available():
    from easycv.datasets.shared.dali_tfrecord_imagenet import DaliLoaderWrapper


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        mode (str): model forward mode
        flush_buffer (bool): flush log buffer
    """

    def __init__(self,
                 dataloader,
                 initial=False,
                 interval=1,
                 mode='test',
                 flush_buffer=True,
                 **eval_kwargs):

        if torch.cuda.is_available():
            if not isinstance(dataloader, DataLoader) and not isinstance(
                    dataloader, DaliLoaderWrapper):
                raise TypeError(
                    'dataloader must be a pytorch DataLoader, but got'
                    f' {type(dataloader)}')
        else:
            if not isinstance(dataloader, DataLoader):
                raise TypeError(
                    'dataloader must be a pytorch DataLoader, but got'
                    f' {type(dataloader)}')

        self.dataloader = dataloader
        self.interval = interval
        self.initial = initial

        self.mode = mode
        self.eval_kwargs = eval_kwargs
        # hook.evaluate runs every interval epoch or iter, popped at init
        self.vis_config = self.eval_kwargs.pop('visualization_config', {})
        self.flush_buffer = flush_buffer

    def before_run(self, runner):
        if self.initial:
            self.after_train_epoch(runner)
        return

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from easycv.apis import single_gpu_test
        if runner.rank == 0:
            if hasattr(runner, 'ema'):
                results = single_gpu_test(
                    runner.ema.model,
                    self.dataloader,
                    mode=self.mode,
                    show=False)
            else:
                results = single_gpu_test(
                    runner.model, self.dataloader, mode=self.mode, show=False)
            self.evaluate(runner, results)

    def add_visualization_info(self, runner, results):
        if runner.visualization_buffer.output.get('eval_results',
                                                  None) is None:
            runner.visualization_buffer.output['eval_results'] = OrderedDict()

        if isinstance(self.dataloader, DataLoader):
            dataset_obj = self.dataloader.dataset
        else:
            dataset_obj = self.dataloader

        if hasattr(dataset_obj, 'visualize'):
            runner.visualization_buffer.output['eval_results'].update(
                dataset_obj.visualize(results, **self.vis_config))

    def evaluate(self, runner, results):
        for _hook in runner.hooks:
            # Only apply `add_visualization_info` when visualization hook is specified
            if isinstance(_hook, (TensorboardLoggerHookV2, WandbLoggerHookV2)):
                self.add_visualization_info(runner, results)
                break

        if isinstance(self.dataloader, DataLoader):
            eval_res = self.dataloader.dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs)
        else:
            eval_res = self.dataloader.evaluate(
                results, logger=runner.logger, **self.eval_kwargs)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        if self.flush_buffer:
            runner.log_buffer.ready = True

        # regist eval_res to do save best hook
        if getattr(runner, 'eval_res', None) is None:
            runner.eval_res = {}

        for k in eval_res.keys():
            tmp_res = {}
            tmp_res[k] = eval_res[k]
            tmp_res['runner_epoch'] = runner.epoch

            if k not in runner.eval_res.keys():
                runner.eval_res[k] = []

            runner.eval_res[k].append(tmp_res)


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        mode (str): model forward mode
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 mode='test',
                 initial=False,
                 gpu_collect=False,
                 flush_buffer=True,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader) and not isinstance(
                dataloader, DaliLoaderWrapper):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.mode = mode
        self.eval_kwargs = eval_kwargs
        self.initial = initial
        self.flush_buffer = flush_buffer
        # hook.evaluate runs every interval epoch or iter, popped at init
        self.vis_config = self.eval_kwargs.pop('visualization_config', {})
        self.gpu_collect = self.eval_kwargs.pop('gpu_collect', gpu_collect)

    def before_run(self, runner):
        if self.initial:
            self.after_train_epoch(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from easycv.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            mode=self.mode,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)

        if runner.rank == 0:
            self.evaluate(runner, results)
