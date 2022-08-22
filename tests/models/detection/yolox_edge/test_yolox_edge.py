# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

from easycv.models.detection.detectors.yolox_edge.yolox_edge import YOLOX_EDGE


class YOLOXEDGETest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_yolox_edge(self):
        for model_type in ['s', 'm', 'l', 'x', 'tiny', 'nano']:
            model = YOLOX_EDGE(
                num_classes=2,
                model_type=model_type,  # s m l x tiny nano
                test_conf=0.01,
                nms_thre=0.65,
                backbone='CSPDarknet',
                head=dict(
                    type='YOLOXHead',
                    model_type=model_type,
                    num_classes=2,
                    stage='EDGE',
                ),
            )
            model = model.cuda()
            model.train()

            batch_size = 2
            imgs = torch.randn(batch_size, 3, 640, 640).cuda()
            num_boxes = 5
            gt_bboxes = torch.randint(
                0, 600, size=(batch_size, num_boxes, 4)).cuda()
            gt_labels = torch.randint(
                0, 1, size=(batch_size, num_boxes, 1)).cuda()
            img_metas = [{'img_shape': (640, 640, 3)}] * batch_size
            kwargs = {
                'gt_bboxes': gt_bboxes,
                'gt_labels': gt_labels,
                'img_metas': img_metas
            }
            output = model(imgs, mode='train', **kwargs)
            self.assertEqual(output['img_h'].cpu().numpy(),
                             np.array(640, dtype=np.float))
            self.assertEqual(output['img_w'].cpu().numpy(),
                             np.array(640, dtype=np.float))
            self.assertEqual(output['total_loss'].shape, torch.Size([]))
            self.assertEqual(output['iou_l'].shape, torch.Size([]))
            self.assertEqual(output['conf_l'].shape, torch.Size([]))
            self.assertEqual(output['cls_l'].shape, torch.Size([]))


if __name__ == '__main__':
    unittest.main()
