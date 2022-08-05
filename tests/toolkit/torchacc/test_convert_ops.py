# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch
from mmcv.runner import init_dist
from torch import distributed as dist
from torch.distributed import ReduceOp

from easycv.utils.test_util import DistributedTestCase

# ReduceOp arg can not be passed through shell cmd, map with str
op_map = {
    'MAX': ReduceOp.MAX,
    'MIN': ReduceOp.MIN,
    'SUM': ReduceOp.SUM,
    'PRODUCT': ReduceOp.PRODUCT
}


def _init_dist(torchacc_enabled):
    if torchacc_enabled:
        from easycv.toolkit.torchacc import torchacc_init
        torchacc_init()
    else:
        init_dist(launcher='pytorch')


def _check_type(module,
                is_raw_module=True,
                torchacc_enabled=False,
                value=None):
    rank = dist.get_rank()

    if torchacc_enabled:
        if value is not None:
            import torchacc.torch_xla.core.xla_model as xm
            assert value.device == xm.xla_device()
        if is_raw_module:
            assert module.__module__ == 'torchacc.torch_xla.core.xla_model'
        else:
            assert module.__module__ == 'easycv.toolkit.torchacc.convert_ops'
    else:
        if value is not None:
            cur_device = torch.device('cuda:{}'.format(rank))
            assert value.device == cur_device
        assert module.__module__ == 'torch.distributed.distributed_c10d'


def _create_value(base_value):
    if isinstance(base_value, (int, float)):
        base_value = torch.tensor(base_value).cuda()
    else:
        base_value = torch.Tensor(base_value).cuda()

    rank = dist.get_rank()
    return base_value * (rank + 1)


def _dist_info(torchacc_enabled=False):
    _init_dist(torchacc_enabled)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    _check_type(dist.get_rank, torchacc_enabled=torchacc_enabled)
    _check_type(dist.get_world_size, torchacc_enabled=torchacc_enabled)

    return rank, world_size


def _reduce(base_value, op_str, dst, torchacc_enabled=False):
    _init_dist(torchacc_enabled)

    value = _create_value(base_value)
    dist.reduce(value, int(dst), op=op_map[op_str])
    _check_type(
        dist.reduce,
        is_raw_module=False,
        torchacc_enabled=torchacc_enabled,
        value=value)

    return value.cpu().numpy().tolist()


def _all_gather(base_value, torchacc_enabled=False):
    _init_dist(torchacc_enabled)

    value = _create_value(base_value)

    world_size = dist.get_world_size()
    tensor_list = [
        torch.zeros(value.size(), dtype=value.dtype, device=value.device)
        for _ in range(world_size)
    ]
    dist.all_gather(tensor_list, value)

    _check_type(
        dist.all_gather,
        is_raw_module=False,
        torchacc_enabled=torchacc_enabled,
        value=value)

    return [i.cpu().numpy().tolist() for i in tensor_list]


def _all_reduce(base_value, op_str, torchacc_enabled=False):
    _init_dist(torchacc_enabled)

    value = _create_value(base_value)
    dist.all_reduce(value, op=op_map[op_str])

    _check_type(
        dist.all_reduce,
        is_raw_module=False,
        torchacc_enabled=torchacc_enabled,
        value=value)

    return value.cpu().numpy().tolist()


def _skip():
    torchacc_enabled = True
    try:
        import torchacc
    except:
        torchacc_enabled = False

    return not (torchacc_enabled and torch.cuda.device_count() > 1)


@unittest.skipIf(_skip(), 'distributed unittest for torchacc')
class ConvertOpsTest(DistributedTestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_dist_info(self):

        def assert_callback(res):
            self.assertEqual(list(res[0]), [0, 2])  # rank 0
            self.assertEqual(list(res[1]), [1, 2])  # rank 1

        # test original all_reduce
        self.start_with_torch(
            _dist_info,
            num_gpus=2,
            assert_callback=assert_callback,
            save_all_ranks=True)

        # test torchacc all_reduce
        kwargs = {'torchacc_enabled': True}
        self.start_with_torchacc(
            _dist_info,
            num_gpus=2,
            assert_callback=assert_callback,
            save_all_ranks=True,
            **kwargs)

    @unittest.skipIf(True, 'fix reduce with hang')
    def test_reduce(self):
        # base value will multiply with rank+1, and return all ranks results
        cases = [
            {
                'assert_callback':
                lambda res: self.assertListEqual(res, [3.0, 2.0]),
                'base_value':
                1.0,
                'op_str':
                'SUM',
                'dst':
                0
            },
            {
                'assert_callback': lambda res: self.assertEqual(res, [2, 4]),
                'base_value': 2,
                'op_str': 'MIN',
                'dst': 0
            },
            {
                'assert_callback':
                lambda res: self.assertListEqual(res, [[1.0], [2.0]]),
                'base_value': [1.0],
                'op_str':
                'MAX',
                'dst':
                1
            },
            {
                'assert_callback':
                lambda res: self.assertListEqual(res, [[1.0, 2.0], [2.0, 8.0]]
                                                 ),
                'base_value': [1.0, 2.0],
                'op_str':
                'PRODUCT',
                'dst':
                1
            },
            {
                'assert_callback':
                lambda res: self.assertListEqual(
                    res,
                    np.asarray([[[[3.0, 6.0], [9.0, 12.0]],
                                 [[12.0, 9.0], [6.0, 3.0]]],
                                [[[2.0, 4.0], [6.0, 8.0]],
                                 [[8.0, 6.0], [4.0, 2.0]]]]).tolist()),
                'base_value': [[[1.0, 2.0], [3.0, 4.0]],
                               [[4.0, 3.0], [2.0, 1.0]]],
                'op_str':
                'SUM',
                'dst':
                0
            },
        ]

        for case in cases:
            # test original all_reduce
            self.start_with_torch(
                _reduce,
                num_gpus=2,
                assert_callback=case['assert_callback'],
                save_all_ranks=True,
                base_value=case['base_value'],
                op_str=case['op_str'],
                dst=case['dst'])

            # test torchacc all_reduce
            kwargs = {'torchacc_enabled': True}
            self.start_with_torchacc(
                _reduce,
                num_gpus=2,
                assert_callback=case['assert_callback'],
                save_all_ranks=True,
                base_value=case['base_value'],
                op_str=case['op_str'],
                dst=case['dst'],
                **kwargs)

    def test_broadcast(self):
        # Not implemented for torchacc yet
        pass

    def test_all_gather(self):
        # base value will multiply with rank+1
        cases = [
            # nor support scalar type
            # {
            #     'assert_callback': lambda res: self.assertEqual(res, [1.0, 2.0]),
            #     'base_value': 1.0,
            # },
            # {
            #     'assert_callback': lambda res: self.assertEqual(res, [2, 4]),
            #     'base_value': 2,
            # },
            {
                'assert_callback':
                lambda res: self.assertListEqual(res, [[1.0], [2.]]),
                'base_value': [1.0],
            },
            {
                'assert_callback':
                lambda res: self.assertListEqual(res, [[1.0, 2.0], [2.0, 4.0]]
                                                 ),
                'base_value': [1.0, 2.0],
            },
            {
                'assert_callback':
                lambda res: self.assertListEqual(
                    res,
                    np.asarray([[[[1.0, 2.0], [3.0, 4.0]],
                                 [[4.0, 3.0], [2.0, 1.0]]],
                                [[[2.0, 4.0], [6.0, 8.0]],
                                 [[8.0, 6.0], [4.0, 2.0]]]]).tolist()),
                'base_value': [[[1.0, 2.0], [3.0, 4.0]],
                               [[4.0, 3.0], [2.0, 1.0]]],
            },
        ]

        for case in cases:
            # test original all_reduce
            self.start_with_torch(
                _all_gather,
                num_gpus=2,
                assert_callback=case['assert_callback'],
                base_value=case['base_value'])

            # test torchacc all_reduce
            kwargs = {'torchacc_enabled': True}
            self.start_with_torchacc(
                _all_gather,
                num_gpus=2,
                assert_callback=case['assert_callback'],
                base_value=case['base_value'],
                **kwargs)

    def test_all_reduce(self):
        # base value will multiply with rank+1
        cases = [
            {
                'assert_callback': lambda res: self.assertEqual(res, 3.0),
                'base_value': 1.0,
                'op_str': 'SUM'
            },
            {
                'assert_callback': lambda res: self.assertEqual(res, 2),
                'base_value': 2,
                'op_str': 'MIN'
            },
            {
                'assert_callback': lambda res: self.assertListEqual(res, [2.]),
                'base_value': [1.0],
                'op_str': 'MAX'
            },
            {
                'assert_callback':
                lambda res: self.assertListEqual(res, [2.0, 8.0]),
                'base_value': [1.0, 2.0],
                'op_str':
                'PRODUCT'
            },
            {
                'assert_callback':
                lambda res: self.assertListEqual(
                    res,
                    np.asarray([[[3.0, 6.0], [9.0, 12.0]],
                                [[12.0, 9.0], [6.0, 3.0]]]).tolist()),
                'base_value': [[[1.0, 2.0], [3.0, 4.0]],
                               [[4.0, 3.0], [2.0, 1.0]]],
                'op_str':
                'SUM'
            },
        ]

        for case in cases:
            # test original all_reduce
            self.start_with_torch(
                _all_reduce,
                num_gpus=2,
                assert_callback=case['assert_callback'],
                base_value=case['base_value'],
                op_str=case['op_str'])

            # test torchacc all_reduce
            kwargs = {'torchacc_enabled': True}
            self.start_with_torchacc(
                _all_reduce,
                num_gpus=2,
                assert_callback=case['assert_callback'],
                base_value=case['base_value'],
                op_str=case['op_str'],
                **kwargs)


if __name__ == '__main__':
    unittest.main()
