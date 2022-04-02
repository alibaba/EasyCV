# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from mmcv.runner import Hook
from mmcv.runner.dist_utils import master_only

from easycv.utils.config_tools import validate_export_config
from .registry import HOOKS


@HOOKS.register_module
class ExportHook(Hook):
    """
    export model when training on pai
    """

    def __init__(
        self,
        cfg,
        ckpt_filename_tmpl='epoch_{}.pth',
        export_ckpt_filename_tmpl='epoch_{}_export.pt',
        export_after_each_ckpt=False,
    ):
        """
        Args:
            cfg: config dict
            ckpt_filename_tmpl: checkpoint filename template
        """
        self.cfg = validate_export_config(cfg)
        self.work_dir = cfg.work_dir
        self.ckpt_filename_tmpl = ckpt_filename_tmpl
        self.export_ckpt_filename_tmpl = export_ckpt_filename_tmpl
        self.export_after_each_ckpt = export_after_each_ckpt or cfg.get(
            'export_after_each_ckpt', False)

    def export_model(self, runner, epoch):
        # epoch = runner.epoch
        ckpt_fname = self.ckpt_filename_tmpl.format(epoch)
        export_ckpt_fname = self.export_ckpt_filename_tmpl.format(epoch)
        local_ckpt = os.path.join(self.work_dir, ckpt_fname)
        export_local_ckpt = os.path.join(self.work_dir, export_ckpt_fname)

        if not os.path.exists(local_ckpt):
            runner.logger.warning(f'{local_ckpt} does not exists, skip export')
        else:
            runner.logger.info(f'export {local_ckpt} to {export_local_ckpt}')
            from easycv.apis.export import export
            export(self.cfg, local_ckpt, export_local_ckpt)

    @master_only
    def after_train_iter(self, runner):
        pass

    @master_only
    def after_train_epoch(self, runner):
        # do export after every ckpy is right! should do so!
        if self.export_after_each_ckpt:
            self.export_model(runner, runner.epoch)
        pass

    @master_only
    def after_run(self, runner):
        self.export_model(runner, runner.epoch)
