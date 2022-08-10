# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
from collections import namedtuple

import torch
import torch.distributed
import torchacc.torch_xla.amp as torchacc_amp
import torchacc.torch_xla.amp.syncfree as xla_optim
import torchacc.torch_xla.core.xla_model as xm
from prettytable import PrettyTable
from torch.distributed import ReduceOp

DEFAULT_TAG = 'EasyCV-default-barrier-tag'

OpSpec = namedtuple('OpSpec', ['module', 'name', 'value'])


class OpsCollector(object):

    def __init__(self):
        self._ops = {}

    def add_op(self, op_name, op_spec):
        self._ops[op_name] = op_spec

    def get_op(self, op_name):
        return self._ops.get(op_name, None)

    @property
    def ops(self):
        return self._ops


class OpsCollectorManager(object):

    def __init__(self):
        self._registries = {}

    def get_collector(self, name):
        return self._registries[name]

    @property
    def registies(self):
        return self._registries

    def register(self, name):
        if name not in self._registries:
            self._registries[name] = OpsCollector()


_ops_manager = OpsCollectorManager()

TORCH_MODULE_NAME = 'torch'
TORCHACC_MODULE_NAME = 'torchacc'


def _register_torch_ops():
    global _ops_manager

    module_name = TORCH_MODULE_NAME
    _ops_manager.register(module_name)
    collector = _ops_manager.get_collector(module_name)

    collector.add_op(
        'TO', OpSpec(module=torch.Tensor, name='to', value=torch.Tensor.to))
    collector.add_op(
        'CUDA',
        OpSpec(module=torch.Tensor, name='cuda', value=torch.Tensor.cuda))
    collector.add_op('TENSOR',
                     OpSpec(module=torch, name='tensor', value=torch.tensor))
    collector.add_op('ZEROS',
                     OpSpec(module=torch, name='zeros', value=torch.zeros))
    collector.add_op(
        'GET_RANK',
        OpSpec(
            module=torch.distributed,
            name='get_rank',
            value=torch.distributed.get_rank))
    collector.add_op(
        'GET_WORLD_SIZE',
        OpSpec(
            module=torch.distributed,
            name='get_world_size',
            value=torch.distributed.get_world_size))
    collector.add_op(
        'BARRIER',
        OpSpec(
            module=torch.distributed,
            name='barrier',
            value=torch.distributed.barrier))
    collector.add_op(
        'ALL_REDUCE',
        OpSpec(
            module=torch.distributed,
            name='all_reduce',
            value=torch.distributed.all_reduce))
    collector.add_op(
        'REDUCE',
        OpSpec(
            module=torch.distributed,
            name='reduce',
            value=torch.distributed.reduce))
    collector.add_op(
        'BROADCAST',
        OpSpec(
            module=torch.distributed,
            name='broadcast',
            value=torch.distributed.broadcast))
    collector.add_op(
        'ALL_GATHER',
        OpSpec(
            module=torch.distributed,
            name='all_gather',
            value=torch.distributed.all_gather))
    collector.add_op(
        'IS_INITIALIZED',
        OpSpec(
            module=torch.distributed,
            name='is_initialized',
            value=torch.distributed.is_initialized))
    collector.add_op(
        'ADAM',
        OpSpec(module=torch.optim, name='Adam', value=torch.optim.Adam))
    collector.add_op(
        'ADAMW',
        OpSpec(module=torch.optim, name='AdamW', value=torch.optim.AdamW))
    collector.add_op(
        'SGD', OpSpec(module=torch.optim, name='SGD', value=torch.optim.SGD))
    collector.add_op(
        'GRADSCALER',
        OpSpec(
            module=torch.cuda.amp,
            name='GradScaler',
            value=torch.cuda.amp.GradScaler))

    return collector


def _register_torchacc_ops():
    global _ops_manager

    reduce_op_map = {
        ReduceOp.SUM: xm.REDUCE_SUM,
        ReduceOp.PRODUCT: xm.REDUCE_MUL,
        ReduceOp.MIN: xm.REDUCE_MIN,
        ReduceOp.MAX: xm.REDUCE_MAX,
        ReduceOp.BAND: xm.REDUCE_AND,
        ReduceOp.BOR: xm.REDUCE_OR,
    }

    module_name = TORCHACC_MODULE_NAME
    _ops_manager.register(module_name)
    collector = _ops_manager.get_collector(module_name)

    origin_to = torch.Tensor.to
    origin_tensor = torch.tensor
    origin_zeros = torch.zeros
    torchacc_device = xm.xla_device()

    from typing import Any, Optional, Union
    from torch.types import _int, _bool, _dtype, _device

    def torcacc_is_initialized():
        # TODO: add initialize attribute
        # keep consistent with torch dist behavior
        return xm.xrt_world_size() > 1

    def torchacc_to(self,
                    device: Optional[Union[_device, str]] = None,
                    dtype: Optional[_dtype] = None,
                    non_blocking: _bool = False,
                    copy: _bool = False) -> torch.Tensor:
        if device is not None and str(device).startswith('cuda'):
            device = torchacc_device
        return origin_to(
            self,
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy)

    # must setattr after torchacc_to
    def torchacc_cuda(self,
                      device: Optional[Union[_device, _int, str]] = None,
                      non_blocking: _bool = False) -> torch.Tensor:
        assert torch.cuda.is_available()
        device = torchacc_device
        return self.to(device=device, non_blocking=non_blocking)

    def torchacc_tensor(data: Any,
                        dtype: Optional[_dtype] = None,
                        device: Union[_device, str, None] = None,
                        requires_grad: _bool = False) -> torch.Tensor:
        if str(device).startswith('cuda'):
            device = torchacc_device
        return origin_tensor(
            data=data, dtype=dtype, device=device, requires_grad=requires_grad)

    def torchacc_zeros(*args, device=None, **kwargs):
        if str(device).startswith('cuda'):
            device = torchacc_device
        return origin_zeros(*args, device=device, **kwargs)

    def torchacc_barrier(tag=None, **kwargs):
        if tag is None:
            tag = DEFAULT_TAG
        return xm.rendezvous(tag, **kwargs)

    def torcacc_all_reduce(tensor, op=ReduceOp.SUM, **kwargs):
        if not isinstance(tensor, list):
            return xm.all_reduce(
                reduce_type=reduce_op_map[op], inputs=[tensor], **kwargs)
        else:
            return xm.all_reduce(
                reduce_type=reduce_op_map[op], inputs=tensor, **kwargs)

    def torchacc_reduce(tensor, dst, op=ReduceOp.SUM, **kwargs):
        # if tensor.device.type != 'gpu':
        #     return
        if xm.get_ordinal() != dst:
            tensor = tensor.clone()
            xm.all_reduce(
                reduce_type=reduce_op_map[op], inputs=tensor, **kwargs)
        else:
            xm.all_reduce(
                reduce_type=reduce_op_map[op], inputs=[tensor], **kwargs)

    def torcacc_broadcast(**kwargs):
        raise ValueError('Not support broadcast for torchacc yet!')

    def torcacc_all_gather(tensor_list, tensor, **kwargs):
        if len(tensor.size()) == 0:
            raise ValueError(
                'Not support ``all_gather`` scaler type for torchacc!')

        res = xm.all_gather(value=tensor, dim=0, **kwargs)
        splits = torch.tensor_split(res, len(tensor_list))

        for i in range(len(tensor_list)):
            assert splits[i].size() == tensor.size(
            ), 'mismatch size: {}, {}'.format(splits[i].size(), tensor.size())
            tensor_list[i] = splits[i]
        del splits

    collector.add_op(
        'TO', OpSpec(module=None, name=None,
                     value=torchacc_to))  # without `to` function, module=None
    collector.add_op('CUDA', OpSpec(
        module=None, name=None,
        value=torchacc_cuda))  # without `cuda` function, module=None
    collector.add_op('TENSOR',
                     OpSpec(module=None, name=None, value=torchacc_tensor))
    collector.add_op('ZEROS',
                     OpSpec(module=None, name=None, value=torchacc_zeros))
    collector.add_op(
        'GET_RANK',
        OpSpec(module=xm, name='get_ordinal', value=xm.get_ordinal))
    collector.add_op(
        'GET_WORLD_SIZE',
        OpSpec(module=xm, name='xrt_world_size', value=xm.xrt_world_size))
    collector.add_op(
        'BARRIER',
        OpSpec(module=xm, name='rendezvous', value=torchacc_barrier))
    collector.add_op(
        'ALL_REDUCE',
        OpSpec(module=xm, name='all_reduce', value=torcacc_all_reduce))
    collector.add_op('REDUCE',
                     OpSpec(module=None, name=None, value=torchacc_reduce))
    collector.add_op('BROADCAST',
                     OpSpec(module=None, name=None, value=torcacc_broadcast)
                     )  # without `broadcast` function, module=None
    collector.add_op(
        'ALL_GATHER',
        OpSpec(module=xm, name='all_gather', value=torcacc_all_gather))
    collector.add_op(
        'IS_INITIALIZED',
        OpSpec(module=None, name=None, value=torcacc_is_initialized))
    collector.add_op(
        'ADAM', OpSpec(module=xla_optim, name='Adam', value=xla_optim.Adam))
    collector.add_op(
        'ADAMW', OpSpec(module=xla_optim, name='AdamW', value=xla_optim.AdamW))
    collector.add_op('SGD',
                     OpSpec(module=xla_optim, name='SGD', value=xla_optim.SGD))
    collector.add_op(
        'GRADSCALER',
        OpSpec(
            module=torchacc_amp,
            name='GradScaler',
            value=torchacc_amp.GradScaler))

    return collector


def convert_ops(src_collector, target_collector):
    reset_ops_map = {}
    for src_op_key, src_op in src_collector.ops.items():
        target_op = target_collector.get_op(src_op_key)
        if src_op.module is None or target_op is None:
            logging.info(
                'Skip {}, source op module is None or not find target op!')
            continue

        setattr(src_op.module, src_op.name, target_op.value)

        reset_ops_map.update({
            '{}.{}'.format(src_op.module.__name__, src_op.name):
            '{}{}'.format(
                target_op.module.__name__ +
                '.' if target_op.module is not None else '', target_op.name
                or target_op.value.__name__)
        })

    show = PrettyTable()
    show.field_names = ['source ops', 'replaced ops']
    for k, v in reset_ops_map.items():
        show.add_row([k, v])

    from torch.distributed import get_rank
    if get_rank() == 0:
        print('Replaced ops is as follows:')
        print(show)


def convert_torch_ops_to_torchacc():
    print(
        'For adapt torchacc, we replaced part of torch\'s operators with torchacc\'s operators.'
    )

    src_collector = _register_torch_ops()
    target_collector = _register_torchacc_ops()
    convert_ops(src_collector, target_collector)


def convert_timm_ops():
    import timm

    # TODO: remove it, fix torch.tensor to adapt torch.jit
    if hasattr(timm.models.layers, 'anti_aliasing'):
        from timm.models.layers import anti_aliasing
        _ori_DownsampleJIT = anti_aliasing.DownsampleJIT

        class FixedDownsampleJIT(_ori_DownsampleJIT):
            pass

        setattr(anti_aliasing, 'DownsampleJIT', FixedDownsampleJIT)

    # TODO: remove it, fix torch.cat to support multiple types of arguments
    def _fix_forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        # torch.cat does not support multiple types of arguments
        # ========================my add=========================
        if x.dtype == torch.float16:
            cls_token = cls_token.half()
        # ======================== end===========================

        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat(
                (cls_token, self.dist_token.expand(x.shape[0], -1, -1), x),
                dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    from timm.models.vision_transformer import VisionTransformer
    setattr(VisionTransformer, 'forward_features', _fix_forward_features)

    print(
        'For adapt to torchacc, we have modified some apis of timm. '
        'Please refer to ``easycv.toolkit.torchacc.convert_timm_ops`` for details.'
    )
