# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
import unittest
from copy import deepcopy
from distutils.version import LooseVersion

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR
from torch.testing._internal.common_utils import TestCase


@unittest.skipIf(
    LooseVersion(torch.__version__) < LooseVersion('1.6.0'),
    'skip some test bugs below 1.6.0 ')
class TestOptim(TestCase):
    exact_dtype = True

    def _test_basic_cases_template(self, weight, bias, input, constructor,
                                   scheduler_constructors):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)
        optimizer = constructor(weight, bias)
        schedulers = []
        for scheduler_constructor in scheduler_constructors:
            schedulers.append(scheduler_constructor(optimizer))

        # to check if the optimizer can be printed as a string
        optimizer.__repr__()

        def fn():
            optimizer.zero_grad()
            y = weight.mv(input)
            if y.is_cuda and bias.is_cuda and y.get_device(
            ) != bias.get_device():
                y = y.cuda(bias.get_device())
            loss = (y + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = fn().item()
        for _i in range(200):
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    val_loss = fn()
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            optimizer.step(fn)
        self.assertLess(fn().item(), initial_value)

    def _test_state_dict(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input_cuda if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        for _i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        weight_c = Variable(weight.data.clone(), requires_grad=True)
        bias_c = Variable(bias.data.clone(), requires_grad=True)
        optimizer_c = constructor(weight_c, bias_c)
        fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
        # Load state dict
        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_c.load_state_dict(state_dict_c)
        # Run both optimizations in parallel
        for _i in range(20):
            optimizer.step(fn)
            optimizer_c.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)
        # Make sure state dict is deterministic with equal but not identical parameters
        self.assertEqual(optimizer.state_dict(), optimizer_c.state_dict())
        # Make sure repeated parameters have identical representation in state dict
        optimizer_c.param_groups.extend(optimizer_c.param_groups)
        self.assertEqual(optimizer.state_dict()['param_groups'][-1],
                         optimizer_c.state_dict()['param_groups'][-1])

        # Check that state dict can be loaded even when we cast parameters
        # to a different type and move to a different device.
        if not torch.cuda.is_available():
            return

        input_cuda = Variable(input.data.float().cuda())
        weight_cuda = Variable(weight.data.float().cuda(), requires_grad=True)
        bias_cuda = Variable(bias.data.float().cuda(), requires_grad=True)
        optimizer_cuda = constructor(weight_cuda, bias_cuda)
        fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda,
                                    bias_cuda)

        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_cuda.load_state_dict(state_dict_c)

        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        for _i in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda)

        # validate deepcopy() copies all public attributes
        def getPublicAttr(obj):
            return set(k for k in obj.__dict__ if not k.startswith('_'))

        try:
            self.assertEqual(
                getPublicAttr(optimizer), getPublicAttr(deepcopy(optimizer)))
        except:
            self.assertEqual(
                getPublicAttr(optimizer), getPublicAttr(deepcopy(optimizer)))

    def _test_basic_cases(self,
                          constructor,
                          scheduler_constructors=None,
                          ignore_multidevice=False):
        if scheduler_constructors is None:
            scheduler_constructors = []
        self._test_state_dict(
            torch.randn(10, 5), torch.randn(10), torch.randn(5), constructor)
        # self._test_basic_cases_template(
        #     torch.randn(10, 5),
        #     torch.randn(10),
        #     torch.randn(5),
        #     constructor,
        #     scheduler_constructors
        # )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2)[..., 0],
            torch.randn(10, 2)[..., 0], torch.randn(5), constructor,
            scheduler_constructors)
        # CUDA
        if not torch.cuda.is_available():
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(),
            torch.randn(10).cuda(),
            torch.randn(5).cuda(), constructor, scheduler_constructors)
        # Multi-GPU
        if not torch.cuda.device_count() > 1 or ignore_multidevice:
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(0),
            torch.randn(10).cuda(1),
            torch.randn(5).cuda(0), constructor, scheduler_constructors)

    def _build_params_dict(self, weight, bias, **kwargs):
        return [{'params': [weight]}, dict(params=[bias], **kwargs)]

    def _build_params_dict_single(self, weight, bias, **kwargs):
        return [dict(params=bias, **kwargs)]

    def test_lars(self):
        from easycv.core.optimizer import LARS
        optimizer = LARS

        self._test_basic_cases(
            lambda weight, bias: optimizer([weight, bias], lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(
            self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(
            self._build_params_dict_single(weight, bias, lr=1e-2), lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(
            self._build_params_dict_single(weight, bias, lr=1e-2)))
        self._test_basic_cases(
            lambda weight, bias: optimizer([weight, bias], lr=1e-3),
            [lambda opt: StepLR(opt, gamma=0.9, step_size=10)])
        self._test_basic_cases(
            lambda weight, bias: optimizer([weight, bias], lr=1e-3), [
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                lambda opt: ReduceLROnPlateau(opt)
            ])
        self._test_basic_cases(
            lambda weight, bias: optimizer([weight, bias], lr=1e-3), [
                lambda opt: StepLR(opt, gamma=0.99, step_size=10),
                lambda opt: ExponentialLR(opt, gamma=0.99),
                lambda opt: ReduceLROnPlateau(opt)
            ])
        self._test_basic_cases(lambda weight, bias: optimizer(
            [weight, bias], lr=1e-3, momentum=1))
        self._test_basic_cases(lambda weight, bias: optimizer(
            [weight, bias], lr=1e-3, momentum=1, weight_decay=1))
        self._test_basic_cases(lambda weight, bias: optimizer(
            [weight, bias], nesterov=True, lr=1e-3, momentum=1, weight_decay=1)
                               )
        with self.assertRaisesRegex(ValueError,
                                    'Invalid momentum value: -0.5'):
            optimizer(None, lr=1e-2, momentum=-0.5)

    def test_ranger(self):
        from easycv.core.optimizer import Ranger
        optimizer = Ranger

        self._test_basic_cases(
            lambda weight, bias: optimizer([weight, bias], lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(
            self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(
            self._build_params_dict_single(weight, bias, lr=1e-2), lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(
            self._build_params_dict_single(weight, bias, lr=1e-2)))
        self._test_basic_cases(
            lambda weight, bias: optimizer([weight, bias], lr=1e-3),
            [lambda opt: StepLR(opt, gamma=0.9, step_size=10)])
        self._test_basic_cases(
            lambda weight, bias: optimizer([weight, bias], lr=1e-3), [
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                lambda opt: ReduceLROnPlateau(opt)
            ])
        self._test_basic_cases(
            lambda weight, bias: optimizer([weight, bias], lr=1e-3), [
                lambda opt: StepLR(opt, gamma=0.99, step_size=10),
                lambda opt: ExponentialLR(opt, gamma=0.99),
                lambda opt: ReduceLROnPlateau(opt)
            ])
        self._test_basic_cases(lambda weight, bias: optimizer(
            [weight, bias],
            lr=1e-3,
            alpha=0.5,
        ))
        self._test_basic_cases(lambda weight, bias: optimizer(
            [weight, bias], lr=1e-3, alpha=0.5, weight_decay=1))
        self._test_basic_cases(lambda weight, bias: optimizer(
            [weight, bias], lr=1e-3, alpha=0.5, weight_decay=1))
        with self.assertRaisesRegex(ValueError,
                                    'Invalid slow update rate: -0.5'):
            optimizer(None, lr=1e-2, alpha=-0.5)


if __name__ == '__main__':
    unittest.main()
