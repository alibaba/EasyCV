# Copyright 2019 Alibaba Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Linear module tests."""

import math
import os
import random
import unittest

import numpy as np
import torch

from easycv.toolkit import sailfish


class MockLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data.
  """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_initializer=None,
                 bias_initializer=None,
                 parallel=None):
        super(MockLinear, self).__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.weight = torch.nn.Parameter(
            torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_initializer = weight_initializer
        if weight_initializer is None:
            self.weight_initializer = sailfish.KaimingUniformInitializer(
                math.sqrt(5))
        self.bias_initializer = bias_initializer
        if bias_initializer is None:
            self.bias_initializer = sailfish.BiasUniformInitializer(
                self.in_features)
        self.reset_parameters()
        self.parallel = parallel

    def reset_parameters(self):
        r"""Reset parameters."""
        self.weight_initializer(self.weight)
        if self.bias is not None:
            self.bias_initializer(self.bias)

    def forward(self, features):  # pylint: disable=arguments-differ
        features = features.type(dtype=self.weight.dtype)
        return torch.nn.functional.linear(features, self.weight, self.bias)


def _run_baseline_train_main(gpus_per_worker, num_steps, batch_size,
                             in_features, out_features, bias, lr):
    r"""Run baseline on 1 GPU."""
    torch.manual_seed(42)
    random.seed(42)
    fc = MockLinear(
        in_features,
        out_features,
        bias=bias,
        weight_initializer=sailfish.ZerosInitializer(),
        bias_initializer=sailfish.OnesInitializer()).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(fc.parameters(), lr=lr)
    fc.train()
    criterion.train()

    results = []
    for step in range(num_steps):
        result = {}
        features_list = []
        label_list = []
        for gpu in range(gpus_per_worker):
            torch.manual_seed(42 * step + gpu)
            random.seed(42 * step + gpu)
            features_list.append(torch.randn([batch_size, in_features]).cuda())
            label_list.append(
                torch.as_tensor([
                    random.randint(0, out_features - 1)
                    for _ in range(batch_size)
                ]).cuda())
        features = torch.cat(features_list)
        label = torch.cat(label_list)

        torch.manual_seed(42 * step)
        random.seed(42 * step)
        logits = fc(features)
        loss = criterion(logits, label)
        result['loss'] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result['grads/norm'] = [
            torch.norm(p.grad).item() for p in fc.parameters()
        ]
        results.append(result)
    return results


def _run_mp_train_main(gpu, gpus_per_worker, baseline_steps, num_steps,
                       batch_size, in_features, out_features, bias, lr):
    r"""Run MP and validate results."""
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        'nccl', rank=gpu, world_size=gpus_per_worker)

    torch.manual_seed(42)
    random.seed(42)
    model_parallel = sailfish.ModelParallel(gpu, gpus_per_worker)
    fc = sailfish.Linear(
        in_features,
        out_features,
        bias=bias,
        weight_initializer=sailfish.ZerosInitializer(),
        bias_initializer=sailfish.OnesInitializer(),
        parallel=model_parallel).cuda()
    criterion = sailfish.CrossEntropyLoss(parallel=model_parallel).cuda()
    optimizer = torch.optim.SGD(fc.parameters(), lr=lr)
    fc.train()
    criterion.train()

    for step in range(num_steps):
        torch.manual_seed(42 * step + gpu)
        random.seed(42 * step + gpu)
        features = torch.randn([batch_size, in_features]).cuda()
        features = model_parallel.gather(features)
        label = torch.as_tensor([
            random.randint(0, out_features - 1) for _ in range(batch_size)
        ]).cuda()
        label = model_parallel.gather_target(label)

        torch.manual_seed(42 * step)
        random.seed(42 * step)
        logits = fc(features)
        loss = criterion(logits, label)
        np.testing.assert_allclose(
            loss.item(),
            baseline_steps[step]['loss'],
            rtol=1e-5,
            err_msg='Wrong loss at gpu {} step {}'.format(gpu, step))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        grad_norms = [
            torch.norm(model_parallel.gather(p.grad)).item()
            for p in fc.parameters()
        ]
        np.testing.assert_allclose(
            grad_norms,
            baseline_steps[step]['grads/norm'],
            rtol=1e-5,
            err_msg='Wrong grads norm at gpu {} step {}'.format(gpu, step))


class TestLinear(unittest.TestCase):
    r"""Test sailfish.Linear."""

    def _run_baseline_train(self, batch_size, in_features, out_features, bias,
                            lr):
        r"""Run baseline without parallel."""
        result = {}
        features = torch.randn([batch_size, in_features])
        fc = torch.nn.Linear(in_features, out_features, bias=bias)
        optimizer = torch.optim.SGD(fc.parameters(), lr=lr)
        fc.train()
        logits = fc(features)
        loss = torch.sum(logits)
        result['loss'] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result['grads/norm'] = [
            torch.norm(p.grad).item() for p in fc.parameters()
        ]
        return result

    def _run_mp_no_parallel_train(self, batch_size, in_features, out_features,
                                  bias, lr):
        r"""Run MP without parallel."""
        result = {}
        features = torch.randn([batch_size, in_features])
        fc = sailfish.Linear(in_features, out_features, bias=bias)
        optimizer = torch.optim.SGD(fc.parameters(), lr=lr)
        fc.train()
        logits = fc(features)
        loss = torch.sum(logits)
        result['loss'] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result['grads/norm'] = [
            torch.norm(p.grad).item() for p in fc.parameters()
        ]
        return result

    def _run_mp_1gpu_train(self, batch_size, in_features, out_features, bias,
                           lr):
        r"""Run MP on 1 GPU."""
        result = {}
        features = torch.randn([batch_size, in_features])
        model_parallel = sailfish.ModelParallel(0, 1)
        fc = sailfish.Linear(
            in_features, out_features, bias=bias, parallel=model_parallel)
        optimizer = torch.optim.SGD(fc.parameters(), lr=lr)
        fc.train()
        logits = fc(features)
        loss = torch.sum(logits)
        result['loss'] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result['grads/norm'] = [
            torch.norm(p.grad).item() for p in fc.parameters()
        ]
        return result

    def test_no_parallel(self):
        r"""Test sailfish.Linear without parallel."""
        batch_size = 3
        in_features = 4
        out_features = 5
        bias = False
        lr = 0.1

        for step in range(5):
            torch.manual_seed(42 + step)
            random.seed(42 + step)
            baseline = self._run_baseline_train(batch_size, in_features,
                                                out_features, bias, lr)

            torch.manual_seed(42 + step)
            random.seed(42 + step)
            rc = self._run_mp_no_parallel_train(batch_size, in_features,
                                                out_features, bias, lr)

            np.testing.assert_allclose(
                rc['loss'], baseline['loss'], err_msg='loss not equal')
            np.testing.assert_allclose(
                rc['grads/norm'],
                baseline['grads/norm'],
                err_msg='norm of grads not equal')

    def test_mp(self):
        r"""Test sailfish.Linear with model parallel on 1 GPU."""
        batch_size = 2
        in_features = 7
        out_features = 4
        bias = True
        lr = 0.6

        for step in range(5):
            torch.manual_seed(100 + step)
            random.seed(100 + step)
            baseline = self._run_baseline_train(batch_size, in_features,
                                                out_features, bias, lr)

            torch.manual_seed(100 + step)
            random.seed(100 + step)
            rc = self._run_mp_1gpu_train(batch_size, in_features, out_features,
                                         bias, lr)

            np.testing.assert_allclose(
                rc['loss'], baseline['loss'], err_msg='loss not equal')
            np.testing.assert_allclose(
                rc['grads/norm'],
                baseline['grads/norm'],
                err_msg='norm of grads not equal')

    def test_mp_vs_1gpu(self):
        r"""Test sailfish.ArcFaceLinear with model parallel."""
        gpus_per_worker = torch.cuda.device_count()
        num_steps = 5
        batch_size = 2
        in_features = 3
        out_features = gpus_per_worker
        bias = True
        lr = 0.6

        baseline_steps = _run_baseline_train_main(gpus_per_worker, num_steps,
                                                  batch_size, in_features,
                                                  out_features, bias, lr)

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '24601'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        torch.multiprocessing.spawn(
            _run_mp_train_main,
            args=(gpus_per_worker, baseline_steps, num_steps, batch_size,
                  in_features, out_features, bias, lr),
            nprocs=gpus_per_worker,
            join=True)


if __name__ == '__main__':
    unittest.main()
