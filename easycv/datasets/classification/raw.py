# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
from PIL import Image

from easycv.core.visualization.image import imshow_label
from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset


@DATASETS.register_module
class ClsDataset(BaseDataset):
    """Dataset for classification

    Args:
        data_source: data source to parse input data
        pipeline: transforms list
    """

    def __init__(self, data_source, pipeline):
        super(ClsDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        results = self.data_source.get_sample(idx)
        img = results['img']
        gt_labels = results['gt_labels']
        if isinstance(img, list):
            # img is list, means contains multi img
            imgs_list = []
            for img_i in img:
                assert isinstance(img_i, Image.Image), \
                    f'The output from the data source must be an Image, got: {type(img_i)}. \
                    Please ensure that the list file does not contain labels.'

                results['img'] = img_i
                img_i = self.pipeline(results)['img'].unsqueeze(0)
                imgs_list.append(img_i)
            results['img'] = torch.cat(imgs_list, dim=0)
            results['gt_labels'] = torch.tensor(gt_labels).long()
        else:
            results = self.pipeline(results)

        return results

    def evaluate(self, results, evaluators, logger=None, topk=(1, 5)):
        '''evaluate classification task

        Args:
            results: a dict of list of tensor, including prediction and groundtruth
                info, where prediction tensor is NxC，and the same with groundtruth labels.

            evaluators: a list of evaluator

        Return:
            eval_result: a dict of float, different metric values
        '''
        assert len(evaluators) == 1, \
            'classification evaluation only support one evaluator'
        gt_labels = results.pop('gt_labels')
        eval_res = evaluators[0].evaluate(results, gt_labels)

        return eval_res

    def visualize(self, results, vis_num=10, **kwargs):
        """Visulaize the model output on validation data.
        Args:
            results: A dictionary containing
                class: List of length number of test images.
                img_metas: List of length number of test images,
                    dict of image meta info, containing filename, img_shape,
                    origin_img_shape and so on.
            vis_num: number of images visualized
        Returns: A dictionary containing
            images: Visulaized images, list of np.ndarray.
            img_metas: List of length number of test images,
                    dict of image meta info, containing filename, img_shape,
                    origin_img_shape and so on.
        """
        vis_imgs = []

        # TODO: support img_metas for torch.jit
        if results.get('img_metas', None) is None:
            return {}

        img_metas = results['img_metas'][:vis_num]
        labels = results['class']

        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']

            vis_img = imshow_label(img=filename, labels=labels, show=False)
            vis_imgs.append(vis_img)

        output = {'images': vis_imgs, 'img_metas': img_metas}

        return output
