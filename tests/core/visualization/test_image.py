# Copyright (c) Alibaba, Inc. and its affiliates.
import tempfile
import unittest

import numpy as np

from easycv.core.visualization import (imshow_bboxes, imshow_keypoints,
                                       imshow_label)


class ImshowTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_imshow_keypoints_2d(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = np.array([[1, 1, 1], [2, 2, 1], [4, 4, 1], [8, 8, 1]],
                        dtype=np.float32)
        pose_result = [kpts]
        skeleton = [[0, 1], [1, 2], [2, 3]]
        pose_kpt_color = [(127, 127, 127)] * len(kpts)
        pose_link_color = [(127, 127, 127)] * len(skeleton)
        _ = imshow_keypoints(
            img,
            pose_result,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            show_keypoint_weight=True)

    def test_imshow_bbox(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[10, 10, 30, 30], [10, 50, 30, 80]],
                          dtype=np.float32)
        labels = ['标签 1', 'label 2']
        colors = ['red', 'green']

        with tempfile.TemporaryDirectory() as tmpdir:
            _ = imshow_bboxes(
                img,
                bboxes,
                labels=labels,
                colors=colors,
                show=False,
                out_file=f'{tmpdir}/out.png')

            # test case of empty bboxes
            _ = imshow_bboxes(
                img,
                np.zeros((0, 4), dtype=np.float32),
                labels=None,
                colors='red',
                show=False)

            # test unmatched bboxes and labels
            try:
                _ = imshow_bboxes(
                    img,
                    np.zeros((0, 4), dtype=np.float32),
                    labels=labels[:1],
                    colors='red',
                    show=False)
            except AssertionError as e:
                self.assertEqual(type(e), AssertionError)
            else:
                self.fail('ValueError not raised')

    def test_imshow_label(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        labels = ['标签 1', 'label 2']

        with tempfile.TemporaryDirectory() as tmpdir:
            _ = imshow_label(
                img=img,
                labels=labels,
                show=False,
                out_file=f'{tmpdir}/out.png')


if __name__ == '__main__':
    unittest.main()
