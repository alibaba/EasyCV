# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.cnn.utils import initialize
from torch import Tensor

from easycv.framework.errors import NotImplementedError, TypeError
from easycv.utils.logger import print_log


class BaseModel(nn.Module, metaclass=ABCMeta):
    ''' base class for model. '''

    def __init__(self, init_cfg=None):
        super(BaseModel, self).__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self) -> bool:
        return self._is_init

    def init_weights(self):
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}')
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of the pre-trained model from being overwritten by the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        logging.warning(
                            'Skip `init_cfg` with `Pretrained` type!')
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

    @abstractmethod
    def forward_train(self, img: Tensor, **kwargs) -> Dict[str, Tensor]:
        """ Abstract interface for model forward in training

        Args:
            img (Tensor): image tensor
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    def forward_test(self, img: Tensor, **kwargs) -> Dict[str, Tensor]:
        """ Abstract interface for model forward in testing

        Args:
            img (Tensor): image tensor
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        raise NotImplementedError

    def forward(self, mode='train', *args, **kwargs):
        if mode == 'train':
            return self.forward_train(*args, **kwargs)
        elif mode == 'test':
            return self.forward_test(*args, **kwargs)

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the \
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data, mode='train')
        loss, log_vars = self._parse_losses(losses)

        if type(data['img']) == list:
            num_samples = len(data['img'][0])
        else:
            num_samples = len(data['img'].data)

        return dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data, mode='train')
        loss, log_vars = self._parse_losses(losses)
        return dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, float):
                log_vars[loss_name] = loss_value
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if not isinstance(loss_value, float):
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
            # for adapt torchacc, returns the original tensor value, because value.item() is very time-consuming,
            # value.item() operation will be executed every log internal frequency,
            # so the bigger the log internal, the better.
            log_vars[loss_name] = loss_value

        return loss, log_vars

    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError
