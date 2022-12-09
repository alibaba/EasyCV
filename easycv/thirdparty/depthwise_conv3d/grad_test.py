import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import dtype2prec_DONTUSE

from depthwise_conv3d import DepthwiseConv3d


class TestConv(TestCase):
    def test_Conv3d_depthwise_naive_groups_cuda(self, dtype=torch.float):
        for depth_multiplier in [1, 2]:
            m = DepthwiseConv3d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to("cuda", dtype)
            i = torch.randn(2, 2, 6, 6, 6, device="cuda", dtype=dtype).div_(2).requires_grad_()
            output = m(i)
            grad_output = torch.randn(2, 2 * depth_multiplier, 4, 4, 4, device="cuda", dtype=dtype) / 2
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = DepthwiseConv3d(1, 1 * depth_multiplier, kernel_size=3).to("cuda", dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = DepthwiseConv3d(1, 1 * depth_multiplier, kernel_size=3).to("cuda", dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.bias.grad.data,
                             torch.cat([m1.bias.grad.data,
                                        m2.bias.grad.data], 0),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data,
                                        m2.weight.grad.data], 0),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)


if __name__ == '__main__':
    test = TestConv()
    # test.test_Conv3d_depthwise_naive_groups_cuda()
    test.grad_check()
