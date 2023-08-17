# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

from easycv.models.backbones import ResNet
from easycv.models.backbones.resnet_jit import ResNetJIT
from easycv.utils.profiling import benchmark_torch_function


class ResnetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_resnet_jit(self):
        with torch.no_grad():
            # input data
            batch_size = 1
            a = torch.rand(batch_size, 3, 224, 224).to('cuda')

            r50 = ResNet(50, out_indices=[4]).to('cuda')
            r50.init_weights()
            r50.eval()

            r50_trace = torch.jit.trace(r50, a).to('cuda')
            r50_trace.eval()

            r50_jittable = ResNetJIT(50, out_indices=[4]).to('cuda')
            r50_jittable.init_weights()
            r50_jittable.eval()
            r50_script = torch.jit.script(r50_jittable, a).to('cuda')
            r50_script.eval()

            # for np1, np2 in zip(r50.named_parameters(), r502.named_parameters()):
            #   n1, p1 = np1
            #   n2, p2 = np2
            #   if p1.size() != p2.size():
            #     print(n1, n2, 'shape not the same')
            #   elif not np.allclose(p1.cpu().data.numpy(), p2.cpu().data.numpy()):
            #     print(n1, n2, 'value not the same')
            # exit(0)

            self.assertTrue(
                np.allclose(
                    r50(a)[-1].cpu().data.numpy(),
                    r50_trace(a)[-1].cpu().data.numpy(),
                    atol=1e-2))
            self.assertTrue(
                np.allclose(
                    r50_jittable(a)[-1].cpu().data.numpy(),
                    r50_script(a)[-1].cpu().data.numpy(),
                    atol=1e-2))

            r50(a)
            iter = 100
            t = benchmark_torch_function(iter, r50, a)
            print(f'origin: {t/batch_size} s/per image')

            t = benchmark_torch_function(iter, r50_trace, a)
            print(f'trace r50: {t/batch_size} s/per image')

            t = benchmark_torch_function(iter, r50_script, a)
            print(f'script r50: {t/batch_size} s/per image')
        # for name, param in r50_trace.named_parameters():
        #   print(name)

        # result
        #  no jit: 0.001548142358660698 s/per image
        # jit trace: 0.0016211424767971039 s/per image
        # jit script: 0.0016227740794420241 s/per image

    @torch.no_grad()
    def test_vision_resnet(self):
        from torchvision import models
        batch_size = 1
        a = torch.rand(batch_size, 3, 224, 224).to('cuda')

        r50 = models.resnet50().to('cuda')
        r50_trace = torch.jit.trace(r50, (a))
        r50_script = torch.jit.script(r50)

        iter = 100

        r50_trace(a)
        t = benchmark_torch_function(iter, r50_trace, a)
        print(f'jit trace: {t/batch_size} s/per image')

        r50_script(a)
        t = benchmark_torch_function(iter, r50_script, a)
        print(f'jit script: {t/batch_size} s/per image')

        r50(a)
        t = benchmark_torch_function(iter, r50, a)
        print(f'origin: {t/batch_size} s/per image')


if __name__ == '__main__':
    unittest.main()
