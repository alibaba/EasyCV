# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os

from mmcv.runner import Hook
from mmcv.runner.dist_utils import master_only

from easycv.file import io
from .registry import HOOKS


@HOOKS.register_module
class OSSSyncHook(Hook):
    """
    upload log files and checkpoints to oss when training on pai
    """

    def __init__(self,
                 work_dir,
                 oss_work_dir,
                 interval=1,
                 ckpt_filename_tmpl='epoch_{}.pth',
                 export_ckpt_filename_tmpl='epoch_{}_export.pt',
                 other_file_list=[],
                 iter_interval=None):
        """
        Args:
            work_dir: work_dir in cfg
            oss_work_dir:  oss directory where to upload local files in work_dir
            interval: upload frequency
            ckpt_filename_tmpl: checkpoint filename template
            other_file_list: other file need to be upload to oss
            iter_interval: upload frequency by iter interval, default to be None, means do it with certain assignment
        """
        self.work_dir = work_dir
        self.oss_work_dir = oss_work_dir
        self.interval = interval
        self.ckpt_filename_tmpl = ckpt_filename_tmpl
        self.export_ckpt_filename_tmpl = export_ckpt_filename_tmpl
        self.other_file_list = other_file_list
        self.iter_interval = iter_interval

    def upload_file(self, runner):
        if hasattr(runner, 'file_upload_perepoch'):
            up_load_file_list = runner.file_upload_perepoch + self.other_file_list
        else:
            up_load_file_list = self.other_file_list

        up_load_file_list = list(set(up_load_file_list))
        epoch = runner.epoch + 1

        # try to up load ckpt model
        ckpt_fname = self.ckpt_filename_tmpl.format(epoch)
        local_ckpt = os.path.join(self.work_dir, ckpt_fname)
        oss_ckpt = os.path.join(self.oss_work_dir, ckpt_fname)
        if not os.path.exists(local_ckpt):
            runner.logger.warning(f'{local_ckpt} does not exists, skip upload')
        else:
            runner.logger.info(f'upload {local_ckpt} to {oss_ckpt}')
            io.safe_copy(local_ckpt, oss_ckpt)

        for other_file in up_load_file_list:
            local_files = glob.glob(
                os.path.join(self.work_dir, other_file), recursive=True)
            for local_file in local_files:
                rel_path = os.path.relpath(local_file, self.work_dir)
                oss_file = os.path.join(self.oss_work_dir, rel_path)
                runner.logger.info(f'upload {up_load_file_list}')
                io.safe_copy(local_file, oss_file)

        # local_tf_logs = os.path.join(self.work_dir, 'tf_logs')
        # oss_tf_logs = os.path.join(self.oss_work_dir, 'tf_logs')
        # runner.logger.info(f'upload directory {local_tf_logs} to {oss_tf_logs}')
        # io.copytree(local_tf_logs, oss_tf_logs)

    # we still use oss sdk to upload pth, log, by default iter 1000, which
    @master_only
    def after_train_iter(self, runner):
        # upload checkpoint and tf events
        if self.iter_interval is not None:
            if not self.every_n_inner_iters(runner, self.iter_interval):
                return
            self.upload_file(runner)
        return

    @master_only
    def after_train_epoch(self, runner):
        # upload checkpoint and tf events
        if not self.every_n_epochs(runner, self.interval):
            return
        self.upload_file(runner)

    @master_only
    def after_run(self, runner):
        # upload final log files
        timestamp = runner.timestamp
        upload_files = [
            '{}.log'.format(timestamp),
            '{}.log.json'.format(timestamp),
        ]
        for log_file in upload_files:
            local_log = os.path.join(self.work_dir, log_file)
            if not os.path.exists(local_log):
                runner.logger.warning(
                    f'{local_log} does not exists, skip upload')
                continue
            oss_log = os.path.join(self.oss_work_dir, log_file)
            runner.logger.info(f'upload {local_log} to {oss_log}')
            io.safe_copy(local_log, oss_log)

        # try to upload exported model
        epoch = runner.epoch
        export_ckpt_fname = self.export_ckpt_filename_tmpl.format(epoch)
        export_local_ckpt = os.path.join(self.work_dir, export_ckpt_fname)
        export_oss_ckpt = os.path.join(self.oss_work_dir, export_ckpt_fname)
        if not os.path.exists(export_local_ckpt):
            runner.logger.warning(
                f'{export_local_ckpt} does not exists, skip upload')
        else:
            runner.logger.info(
                f'upload {export_local_ckpt} to {export_oss_ckpt}')
            io.safe_copy(export_local_ckpt, export_oss_ckpt)
