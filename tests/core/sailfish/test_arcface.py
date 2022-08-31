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
"""ArcFaceLinear module tests."""

import os
import random
import unittest

import numpy as np
import torch

from easycv.toolkit import sailfish


def mp_vs_ddp_main(gpu, gpus_per_worker):
    r"""Model parallel vs. DDP"""
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        'nccl', rank=gpu, world_size=gpus_per_worker)
    try:
        num_steps = 5
        freeze_num_steps = 5
        learning_rate = 0.1
        batch_size = 2
        image_size = 5
        emb_size = 3
        num_classes = 8
        margin_m = 0.5
        margin_s = 64
        momentum = 0.9
        model_parallel = sailfish.ModelParallel(gpu, gpus_per_worker)

        zeros_init = sailfish.ZerosInitializer()

        # baseline
        torch.manual_seed(42)
        random.seed(42)
        baseline_fe = torch.nn.Linear(image_size, emb_size).cuda()
        baseline_fe = torch.nn.parallel.DistributedDataParallel(
            baseline_fe, device_ids=[gpu])
        baseline_fe_params = list(baseline_fe.parameters())
        baseline_fc = sailfish.ArcFaceLinear(
            emb_size,
            num_classes,
            margin=margin_m,
            scale=margin_s,
            weight_initializer=zeros_init).cuda()
        baseline_fc = torch.nn.parallel.DistributedDataParallel(
            baseline_fc, device_ids=[gpu])
        baseline_fc_params = list(baseline_fc.parameters())
        baseline_criterion = torch.nn.CrossEntropyLoss().cuda()
        baseline_optimizer = torch.optim.SGD(
            [{
                'params': baseline_fe.parameters()
            }, {
                'params': baseline_fc.parameters()
            }],
            lr=learning_rate,
            momentum=momentum)
        baseline_fe.train()
        baseline_fc.train()
        baseline_criterion.train()

        # hybrid parallelism
        torch.manual_seed(42)
        random.seed(42)
        fe = torch.nn.Linear(image_size, emb_size).cuda()
        fe = torch.nn.parallel.DistributedDataParallel(fe, device_ids=[gpu])
        fe_params = list(fe.parameters())
        fc = sailfish.ArcFaceLinear(
            emb_size,
            num_classes,
            margin=margin_m,
            scale=margin_s,
            weight_initializer=zeros_init,
            parallel=model_parallel).cuda()
        fc_params = list(fc.parameters())
        criterion = sailfish.CrossEntropyLoss(parallel=model_parallel).cuda()
        optimizer = torch.optim.SGD([{
            'params': fe.parameters()
        }, {
            'params': fc.parameters()
        }],
                                    lr=learning_rate,
                                    momentum=momentum)
        fe.train()
        fc.train()
        criterion.train()

        for step in range(num_steps):
            # baseline
            torch.manual_seed(42 * step + gpu)
            random.seed(42 * step + gpu)
            baseline_data = torch.randn([batch_size, image_size]).cuda()
            baseline_label = torch.as_tensor([
                random.randint(0, num_classes - 1) for _ in range(batch_size)
            ]).cuda()
            baseline_features = baseline_fe(baseline_data)
            baseline_logits = baseline_fc(baseline_features, baseline_label)
            baseline_loss = baseline_criterion(baseline_logits, baseline_label)
            baseline_loss = model_parallel.reduce_sum(baseline_loss)
            baseline_loss = baseline_loss / gpus_per_worker
            baseline_optimizer.zero_grad()
            baseline_loss.backward()
            baseline_optimizer.step()

            # hybrid parallelism
            torch.manual_seed(42 * step + gpu)
            random.seed(42 * step + gpu)
            data = torch.randn([batch_size, image_size]).cuda()
            label = torch.as_tensor([
                random.randint(0, num_classes - 1) for _ in range(batch_size)
            ]).cuda()
            features = fe(data)
            all_features = model_parallel.gather(features)
            all_label = model_parallel.gather_target(label)
            shard_logits = fc(all_features, all_label)
            loss = criterion(shard_logits, all_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # eval
            torch.manual_seed(42 * step + gpu)
            random.seed(42 * step + gpu)
            with torch.no_grad():
                gathered_logits = model_parallel.gather(shard_logits, dim=1)
                gathered_baseline_logits = model_parallel.gather(
                    baseline_logits, dim=0)
                logits_norm_val = torch.norm(gathered_logits).item()
                baseline_logits_norm_val = torch.norm(
                    gathered_baseline_logits).item()
                np.testing.assert_allclose(
                    logits_norm_val,
                    baseline_logits_norm_val,
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg='logits at gpu {} step {}'.format(gpu, step))

                loss_val = loss.cpu().detach().numpy()
                baseline_loss_val = baseline_loss.cpu().detach().numpy()
                np.testing.assert_allclose(
                    loss_val,
                    baseline_loss_val,
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg='loss at gpu {} step {}'.format(gpu, step))

                fc_grad = model_parallel.gather(fc_params[0].grad)
                baseline_fc_grad = baseline_fc_params[0].grad
                np.testing.assert_allclose(
                    fc_grad.cpu().detach().numpy(),
                    baseline_fc_grad.cpu().detach().numpy(),
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg='fc grad at gpu {} step {}'.format(gpu, step))

                fe_weight = fe_params[0]
                baseline_fe_weight = baseline_fe_params[0]
                np.testing.assert_allclose(
                    fe_weight.cpu().detach().numpy(),
                    baseline_fe_weight.cpu().detach().numpy(),
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg='fe weight at gpu {} step {}'.format(gpu, step))

        for p in baseline_fe.parameters():
            p.requires_grad = False
        for p in fe.parameters():
            p.requires_grad = False
        for step in range(freeze_num_steps):
            # baseline
            torch.manual_seed(100 * step + gpu)
            random.seed(100 * step + gpu)
            baseline_data = torch.randn([batch_size, image_size]).cuda()
            baseline_label = torch.as_tensor([
                random.randint(0, num_classes - 1) for _ in range(batch_size)
            ]).cuda()
            baseline_features = baseline_fe(baseline_data)
            baseline_logits = baseline_fc(baseline_features, baseline_label)
            baseline_loss = baseline_criterion(baseline_logits, baseline_label)
            baseline_loss = model_parallel.reduce_sum(baseline_loss)
            baseline_loss = baseline_loss / gpus_per_worker
            baseline_optimizer.zero_grad()
            baseline_loss.backward()
            baseline_optimizer.step()

            # hybrid parallelism
            torch.manual_seed(100 * step + gpu)
            random.seed(100 * step + gpu)
            data = torch.randn([batch_size, image_size]).cuda()
            label = torch.as_tensor([
                random.randint(0, num_classes - 1) for _ in range(batch_size)
            ]).cuda()
            features = fe(data)
            all_features = model_parallel.gather(features)
            all_label = model_parallel.gather_target(label)
            shard_logits = fc(all_features, all_label)
            loss = criterion(shard_logits, all_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # eval
            torch.manual_seed(100 * step + gpu)
            random.seed(100 * step + gpu)
            with torch.no_grad():
                gathered_logits = model_parallel.gather(shard_logits, dim=1)
                gathered_baseline_logits = model_parallel.gather(
                    baseline_logits, dim=0)
                logits_norm_val = torch.norm(gathered_logits).item()
                baseline_logits_norm_val = torch.norm(
                    gathered_baseline_logits).item()
                np.testing.assert_allclose(
                    logits_norm_val,
                    baseline_logits_norm_val,
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg='freeze logits at gpu {} step {}'.format(
                        gpu, step))

                loss_val = loss.cpu().detach().numpy()
                baseline_loss_val = baseline_loss.cpu().detach().numpy()
                np.testing.assert_allclose(
                    loss_val,
                    baseline_loss_val,
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg='freeze loss at gpu {} step {}'.format(gpu, step))

                fc_grad = model_parallel.gather(fc_params[0].grad)
                baseline_fc_grad = baseline_fc_params[0].grad
                np.testing.assert_allclose(
                    fc_grad.cpu().detach().numpy(),
                    baseline_fc_grad.cpu().detach().numpy(),
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg='freeze fc grad at gpu {} step {}'.format(
                        gpu, step))

                fe_weight = fe_params[0]
                baseline_fe_weight = baseline_fe_params[0]
                np.testing.assert_allclose(
                    fe_weight.cpu().detach().numpy(),
                    baseline_fe_weight.cpu().detach().numpy(),
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg='freeze fe weight at gpu {} step {}'.format(
                        gpu, step))

    finally:
        torch.distributed.destroy_process_group()


def mp_main(gpu,
            gpus_per_worker,
            results,
            num_steps=1,
            batch_size=1,
            num_classes=8):
    r"""Model parallel"""
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        'nccl', rank=gpu, world_size=gpus_per_worker)
    zeros_init = sailfish.ZerosInitializer()
    try:
        emb_size = 3
        learning_rate = 0.1
        margin_m = 0.5
        margin_s = 64
        momentum = 0.9
        image_size = 6
        model_parallel = sailfish.ModelParallel(gpu, gpus_per_worker)

        # hybrid parallelism
        torch.manual_seed(42)
        random.seed(42)
        fe = torch.nn.Linear(image_size, emb_size).cuda()
        fc = sailfish.ArcFaceLinear(
            emb_size,
            num_classes,
            margin=margin_m,
            scale=margin_s,
            weight_initializer=zeros_init,
            parallel=model_parallel).cuda()
        fc_params = list(fc.parameters())
        criterion = sailfish.CrossEntropyLoss(parallel=model_parallel).cuda()
        optimizer = torch.optim.SGD(
            fc.parameters(), lr=learning_rate, momentum=momentum)
        fc.train()
        criterion.train()

        for step in range(num_steps):
            baseline = results[step]
            torch.manual_seed(42 * step + gpu)
            random.seed(42 * step + gpu)
            data = torch.randn([batch_size, image_size]).cuda()
            features = fe(data)
            label = torch.as_tensor([
                random.randint(0, num_classes - 1) for _ in range(batch_size)
            ]).cuda()
            all_features = model_parallel.gather(features)
            all_label = model_parallel.gather_target(label)
            torch.manual_seed(42 * step)
            random.seed(42 * step)
            np.testing.assert_equal(
                list(all_features.size()),
                baseline['features/size'],
                err_msg='Wrong features size at gpu {} step {}'.format(
                    gpu, step))
            np.testing.assert_allclose(
                torch.norm(all_features).item(),
                baseline['features/norm'],
                rtol=1e-5,
                err_msg='Wrong features norm at gpu {} step {}'.format(
                    gpu, step))
            shard_logits = fc(all_features, all_label)
            loss = criterion(shard_logits, all_label)
            np.testing.assert_allclose(
                loss.item(),
                baseline['loss'],
                rtol=1e-5,
                err_msg='Wrong loss at gpu {} step {}'.format(gpu, step))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            fc_grad = model_parallel.gather(fc_params[0].grad)
            np.testing.assert_allclose(
                torch.norm(fc_grad).item(),
                baseline['logits/grad/norm'],
                rtol=1e-5,
                err_msg='Wrong logits grad at gpu {} step {}'.format(
                    gpu, step))

    finally:
        torch.distributed.destroy_process_group()


def baseline_main(gpus_per_worker, num_steps=1, batch_size=1, num_classes=8):
    r"""run on 1 GPU"""
    emb_size = 3
    learning_rate = 0.1
    momentum = 0.9
    image_size = 6

    zeros_init = sailfish.ZerosInitializer()

    # hybrid parallelism
    torch.manual_seed(42)
    random.seed(42)
    fe = torch.nn.Linear(image_size, emb_size).cuda()
    fc = sailfish.ArcFaceLinear(
        emb_size, num_classes, weight_initializer=zeros_init).cuda()
    fc_params = list(fc.parameters())
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        fc.parameters(), lr=learning_rate, momentum=momentum)
    fc.train()
    criterion.train()

    results = []
    for step in range(num_steps):
        result_item = {}
        features_list = []
        label_list = []
        for gpu in range(gpus_per_worker):
            torch.manual_seed(42 * step + gpu)
            random.seed(42 * step + gpu)
            features_list.append(
                fe(torch.randn([batch_size, image_size]).cuda()))
            label_list.append(
                torch.as_tensor([
                    random.randint(0, num_classes - 1)
                    for _ in range(batch_size)
                ]).cuda())
        all_features = torch.cat(features_list)
        all_label = torch.cat(label_list)
        torch.manual_seed(42 * step)
        random.seed(42 * step)
        result_item['features/size'] = list(all_features.size())
        result_item['features/norm'] = torch.norm(all_features).item()
        logits = fc(all_features, all_label)
        loss = criterion(logits, all_label)
        result_item['loss'] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result_item['logits/grad/norm'] = torch.norm(fc_params[0].grad).item()
        results.append(result_item)
    return results


class TestArcFaceLinear(unittest.TestCase):
    r"""Test sailfish.ArcFaceLinear."""

    def test_no_parallel(self):
        r"""Test sailfish.ArcFaceLinear without parallel."""
        in_features = 1
        out_features = 2
        margin_m = random.random()
        margin_s = random.random()

        features = torch.randn([1, in_features])
        label = torch.as_tensor([random.randint(0, out_features - 1)])

        torch.manual_seed(42)
        random.seed(42)
        baseline = sailfish.ArcFaceLinear(
            in_features, out_features, margin=margin_m, scale=margin_s)
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=1.)
        baseline.train()
        baseline_logits = baseline(features, label)
        baseline_loss = torch.sum(baseline_logits)
        baseline_optimizer.zero_grad()
        baseline_loss.backward()
        baseline_optimizer.step()

        torch.manual_seed(42)
        random.seed(42)
        fc = sailfish.ArcFaceLinear(
            in_features, out_features, margin=margin_m, scale=margin_s)
        optimizer = torch.optim.SGD(fc.parameters(), lr=1.)
        fc.train()
        logits = fc(features, label)
        loss = torch.sum(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        np.testing.assert_allclose(
            logits.detach().numpy(),
            baseline_logits.detach().numpy(),
            err_msg='logits not equal to baseline')
        np.testing.assert_allclose(
            [p.detach().numpy() for p in baseline.parameters()],
            [p.detach().numpy() for p in fc.parameters()],
            err_msg='parameters not equal to baseline')

    def test_mp(self):
        r"""Test sailfish.ArcFaceLinear on 1 GPU."""
        in_features = 1
        out_features = 2

        features = torch.randn([1, in_features])
        label = torch.as_tensor([random.randint(0, out_features - 1)])

        torch.manual_seed(42)
        random.seed(42)
        baseline = sailfish.ArcFaceLinear(in_features, out_features)
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=1.)
        baseline.train()
        baseline_logits = baseline(features, label)
        baseline_loss = torch.sum(baseline_logits)
        baseline_optimizer.zero_grad()
        baseline_loss.backward()
        baseline_optimizer.step()

        torch.manual_seed(42)
        random.seed(42)
        model_parallel = sailfish.ModelParallel(0, 1)
        fc = sailfish.ArcFaceLinear(
            in_features, out_features, parallel=model_parallel)
        optimizer = torch.optim.SGD(fc.parameters(), lr=1.)
        fc.train()
        logits = fc(features, label)
        loss = torch.sum(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        np.testing.assert_allclose(
            logits.detach().numpy(),
            baseline_logits.detach().numpy(),
            err_msg='logits not equal to baseline')
        np.testing.assert_allclose(
            [p.detach().numpy() for p in baseline.parameters()],
            [p.detach().numpy() for p in fc.parameters()],
            err_msg='parameters not equal to baseline')

    def cant_test_mp_vs_ddp(self):
        r"""Test sailfish.ArcFaceLinear with model parallel."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '24601'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'

        gpus_per_worker = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            mp_vs_ddp_main,
            args=(gpus_per_worker, ),
            nprocs=gpus_per_worker,
            join=True)

    def test_mp_vs_1gpu(self):
        r"""Test sailfish.ArcFaceLinear with model parallel."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '24601'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'

        gpus_per_worker = torch.cuda.device_count()
        num_steps = 5
        batch_size = 1
        num_classes = gpus_per_worker
        results = baseline_main(gpus_per_worker, num_steps, batch_size,
                                num_classes)
        torch.multiprocessing.spawn(
            mp_main,
            args=(gpus_per_worker, results, num_steps, batch_size,
                  num_classes),
            nprocs=gpus_per_worker,
            join=True)


if __name__ == '__main__':
    unittest.main()
