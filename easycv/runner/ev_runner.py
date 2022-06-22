# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
from distutils.version import LooseVersion

import torch
from mmcv.runner import EpochBasedRunner
from mmcv.runner.log_buffer import LogBuffer

from easycv.file import io
from easycv.utils.checkpoint import load_checkpoint, save_checkpoint

if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
    from torch.cuda import amp


class EVRunner(EpochBasedRunner):

    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 fp16_enable=False):
        """ Epoch Runner for easycv, add support for oss IO and file sync.

        Args:
            model (:obj:`torch.nn.Module`): The model to be run.
            batch_processor (callable): A callable method that process a data
                batch. The interface of this method should be
                `batch_processor(model, data, train_mode) -> dict`
            optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
                optimizer (in most cases) or a dict of optimizers (in models that
                requires more than one optimizer, e.g., GAN).
            work_dir (str, optional): The working directory to save checkpoints
                and logs. Defaults to None.
            logger (:obj:`logging.Logger`): Logger used during training.
                Defaults to None. (The default value is just for backward
                compatibility)
            meta (dict | None): A dict records some import information such as
                environment info and seed, which will be logged in logger hook.
                Defaults to None.
            fp16_enable (bool): if use fp16
        """

        super().__init__(model, batch_processor, optimizer, work_dir, logger,
                         meta)
        self.data_loader = None
        self.fp16_enable = fp16_enable
        self.visualization_buffer = LogBuffer()
        if self.fp16_enable and LooseVersion(
                torch.__version__) < LooseVersion('1.6.0'):
            # convert model to fp16
            self.model.half()
            # patch the normalization layers to make it work in fp32 mode
            from mmcv.runner.fp16_utils import patch_norm_fp32
            patch_norm_fp32(self.model)

    def run_iter(self, data_batch, train_mode, **kwargs):
        """ process for each iteration.

        Args:
            data_batch: Batch of dict of data.
            train_model (bool): If set True, run training step else validation step.
        """
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        """ Training process for one epoch which will iterate through all \
            training data and call hooks at different stages.

        Args:
            data_loader: data loader object for training
        """

        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            # use amp from pytorch 1.6 or later, we should use amp.autocast
            if self.fp16_enable and LooseVersion(
                    torch.__version__) >= LooseVersion('1.6.0'):
                with amp.autocast():
                    self.run_iter(data_batch, train_mode=True)
            else:
                self.run_iter(data_batch, train_mode=True)

            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')

        self._epoch += 1

    def val(self, data_loader, **kwargs):
        """ Validation step which Deprecated, using evaluation hook instead.
        """
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save checkpoint to file.

        Args:
            out_dir: Directory where checkpoint files are to be saved.
            filename_tmpl (str, optional): Checkpoint filename pattern.
            save_optimizer (bool, optional): save optimizer state.
            meta (dict, optional): Metadata to be saved in checkpoint.
        """
        # implement save checkpoint to oss
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = os.path.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        if hasattr(self, 'ema'):
            model = self.ema.model
        else:
            model = self.model

        save_checkpoint(model, filepath, optimizer, meta)

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        # add interface to selfdefine current_lr_fn for lr_hook
        # so that runner can logging correct lrs
        if hasattr(self, 'current_lr_fn'):
            lr = self.current_lr_fn(self.optimizer)
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    # !! notice, map_location should be cpu, other wise may stack to some GPU ,which cause OOMß
    def load_checkpoint(self,
                        filename,
                        map_location=torch.device('cpu'),
                        strict=False,
                        logger=None):
        """Load checkpoint from a file or URL.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``, ``oss://xxx``. Please refer to
                ``docs/source/model_zoo.md`` for details.
            map_location (str): Same as :func:`torch.load`.
            strict (bool): Whether to allow different params for the model and
                checkpoint.
            logger (:mod:`logging.Logger` or None): The logger for error message.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """
        return load_checkpoint(
            self.model,
            filename=filename,
            map_location=map_location,
            strict=strict,
            logger=logger)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        """ Resume state dict from checkpoint.

        Args:
            checkpoint: Checkpoint path
            resume_optimizer: Whether to resume optimizer state
            map_location (str): Same as :func:`torch.load`.

        """

        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)
