# Copyright (c) Alibaba, Inc. and its affiliates.
from mmcv.runner import Hook
from mmcv.runner.dist_utils import master_only

from easycv.utils.logger import get_root_logger
from .registry import HOOKS


@HOOKS.register_module()
class BestCkptSaverHook(Hook):
    """Save checkpoints periodically.

    Args:
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        best_metric_name (List(str)) : metric name to save best, such as "neck_top1"...
            Default: [], do not save anything
        best_metric_type (List(str)) : metric type to define best, should be "max", "min"
            if len(best_metric_type) <= len(best_metric_type), use "max" to append.
    """

    def __init__(self,
                 by_epoch=True,
                 save_optimizer=True,
                 best_metric_name=[],
                 best_metric_type=[],
                 **kwargs):
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = None
        self.best_metric_name = best_metric_name
        self.best_metric_type = best_metric_type

        self.logger = get_root_logger()
        if len(self.best_metric_name) == 0:
            self.logger.warning(
                'BestCkptSaverHook should assign best_metric_name, otherwise ,no best ckpt should will save'
            )
        if len(self.best_metric_name) != len(self.best_metric_type):
            self.logger.warning(
                f'BestCkptSaverHook should have same length of best_metric_name and best_metric_type ({len(self.best_metric_name)} vs {len(self.best_metric_type)})'
            )
            self.logger.warning(
                'BestCkptSaverHook will use max as default metric type')

        while len(self.best_metric_type) < len(self.best_metric_name):
            self.best_metric_type.append('max')

        self.args = kwargs

    @master_only
    def before_run(self, runner):
        if not hasattr(runner, 'file_upload_perepoch'):
            runner.file_upload_perepoch = []

        if not self.out_dir:
            self.out_dir = runner.work_dir

        self.after_train_epoch(runner)

    @master_only
    def after_train_epoch(self, runner):

        if len(self.best_metric_name) > 0 and hasattr(runner, 'eval_res'):
            self.logger.info(f'SaveBest metric_name: {self.best_metric_name}')
            for k in runner.eval_res.keys():
                result_list = runner.eval_res[k]
                if len(result_list) > 0:
                    keys = list(result_list[0].keys())
                    keys.remove('runner_epoch')
                    for key in keys:
                        if key in self.best_metric_name:
                            metric_type = eval(self.best_metric_type[
                                self.best_metric_name.index(key)])
                            maxr = metric_type(
                                result_list, key=lambda x: x[key])
                            if maxr['runner_epoch'] == runner.epoch:
                                runner.file_upload_perepoch.append(
                                    '%s_best.pth' % (key))
                                runner.file_upload_perepoch = list(
                                    set(runner.file_upload_perepoch))
                                meta = {'epoch': runner.epoch - 1, **maxr}
                                runner.save_checkpoint(
                                    self.out_dir,
                                    filename_tmpl='%s_best.pth' % (key),
                                    save_optimizer=self.save_optimizer,
                                    meta=meta)

            self.logger.info('End SaveBest metric')
