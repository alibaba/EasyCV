# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tarfile

import requests
from mmcv.runner import Hook
from mmcv.runner.dist_utils import master_only

from easycv.utils.config_tools import validate_export_config
from .registry import HOOKS


def make_targz(output_filename, source_dir):
    """
    一次性打包目录为tar.gz
    :param output_filename: 压缩文件名
    :param source_dir: 需要打包的目录
    :return: bool
    """
    try:
        with tarfile.open(output_filename, 'w:gz') as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

        return True
    except Exception as e:
        print(e)
        return False


def untar(fname, dirs):
    """
    解压tar.gz文件
    :param fname: 压缩文件名
    :param dirs: 解压后的存放路径
    :return: bool
    """
    try:
        t = tarfile.open(fname)
        t.extractall(path=dirs)
        return True
    except Exception as e:
        print(e)
        return False


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
        export_ckpt_fname = self.export_ckpt_filename_tmpl.format(epoch)
        export_local_ckpt = os.path.join(self.work_dir, export_ckpt_fname)

        runner.logger.info(f'export model to {export_local_ckpt}')
        from easycv.apis.export import export
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model
        export(
            self.cfg,
            ckpt_path='dummy',
            filename=export_local_ckpt,
            model=model)

        origin_tar_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/pai_test/eas_test/easycv/ocr_en.tar.gz'
        r = requests.get(origin_tar_path)
        # download config in current dir
        work_dir = self.work_dir
        origin_targz_path = os.path.join(work_dir,
                                         origin_tar_path.split('/')[-1])
        while not os.path.exists(origin_targz_path):
            try:
                with open(origin_targz_path, 'wb') as code:
                    code.write(r.content)
            except:
                pass
        print('Complete file download!')

        # decompression targz
        untar(origin_targz_path, work_dir)
        print('Complete untar!')

        # finetune model replace origin model
        finetune_model_path = export_local_ckpt
        origin_model_path = os.path.join(
            work_dir,
            os.path.join(
                origin_tar_path.split('/')[-1].split('.')[0],
                'detection/english_det.pth'))
        shutil.copyfile(finetune_model_path, origin_model_path)
        print('Complete copyfile!')

        # compress targz
        finetune_folder_path = os.path.join(
            work_dir,
            origin_tar_path.split('/')[-1].split('.')[0])
        finetune_targz_path = os.path.join(
            work_dir,
            origin_tar_path.split('/')[-1].split('.')[0] + '_finetune.tar.gz')
        make_targz(finetune_targz_path, finetune_folder_path)
        print('Complete make_targz!')

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
