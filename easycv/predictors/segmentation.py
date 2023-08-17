# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import mmcv
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from easycv.core.visualization.image import imshow_bboxes
from easycv.predictors.builder import PREDICTORS
from .base import OutputProcessor, PredictorV2


@PREDICTORS.register_module()
class SegmentationPredictor(PredictorV2):
    """Predictor for Segmentation.

    Args:
        model_path (str): Path of model path.
        config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
        batch_size (int): batch size for forward.
        device (str): Support 'cuda' or 'cpu', if is None, detect device automatically.
        save_results (bool): Whether to save predict results.
        save_path (str): File path for saving results, only valid when `save_results` is True.
        pipelines (list[dict]): Data pipeline configs.
        input_processor_threads (int): Number of processes to process inputs.
        mode (str): The image mode into the model.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 input_processor_threads=8,
                 mode='BGR',
                 *args,
                 **kwargs):

        super(SegmentationPredictor, self).__init__(
            model_path,
            config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            input_processor_threads=input_processor_threads,
            mode=mode,
            *args,
            **kwargs)

        self.CLASSES = self.cfg.CLASSES
        self.PALETTE = self.cfg.get('PALETTE', None)

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """

        img = mmcv.imread(img)
        img = img.copy()
        # seg = result[0]
        seg = result
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            return img


class Mask2formerOutputProcessor(OutputProcessor):
    """Process the output of Mask2former.

    Args:
        task_mode (str): Support task in ['panoptic', 'instance', 'semantic'].
        classes (list): Classes name list.
    """

    def __init__(self, task_mode, classes):
        super(Mask2formerOutputProcessor, self).__init__()
        self.task_mode = task_mode
        self.classes = classes

    def process_single(self, inputs):
        output = {}
        if self.task_mode == 'panoptic':
            pan_results = inputs['pan_results']
            # keep objects ahead
            ids = np.unique(pan_results)[::-1]
            legal_indices = ids != len(self.classes)  # for VOID label
            ids = ids[legal_indices]
            labels = np.array([id % 1000 for id in ids], dtype=np.int64)
            segms = (pan_results[None] == ids[:, None, None])
            masks = [it.astype(np.int32) for it in segms]
            labels_txt = np.array(self.classes)[labels].tolist()

            output['masks'] = masks
            output['labels'] = labels_txt
            output['labels_ids'] = labels
        elif self.task_mode == 'instance':
            output['segms'] = inputs['detection_masks']
            output['bboxes'] = inputs['detection_boxes']
            output['scores'] = inputs['detection_scores']
            output['labels'] = inputs['detection_classes']
        elif self.task_mode == 'semantic':
            output['seg_pred'] = inputs['seg_pred']
        else:
            raise ValueError(f'Not support model {self.task_mode}')
        return output


@PREDICTORS.register_module()
class Mask2formerPredictor(SegmentationPredictor):
    """Predictor for Mask2former.

    Args:
        model_path (str): Path of model path.
        config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
        batch_size (int): batch size for forward.
        device (str): Support 'cuda' or 'cpu', if is None, detect device automatically.
        save_results (bool): Whether to save predict results.
        save_path (str): File path for saving results, only valid when `save_results` is True.
        pipelines (list[dict]): Data pipeline configs.
        input_processor_threads (int): Number of processes to process inputs.
        mode (str): The image mode into the model.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 task_mode='panoptic',
                 input_processor_threads=8,
                 mode='BGR',
                 *args,
                 **kwargs):
        super(Mask2formerPredictor, self).__init__(
            model_path,
            config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            input_processor_threads=input_processor_threads,
            mode=mode,
            *args,
            **kwargs)
        self.task_mode = task_mode
        self.class_name = self.cfg.CLASSES
        self.PALETTE = self.cfg.PALETTE

    def get_output_processor(self):
        return Mask2formerOutputProcessor(self.task_mode, self.CLASSES)

    def model_forward(self, inputs):
        """Model forward.
        """
        with torch.no_grad():
            outputs = self.model.forward(**inputs, mode='test', encode=False)
        return outputs

    def show_panoptic(self, img, masks, labels_ids, **kwargs):
        palette = np.asarray(self.cfg.PALETTE)
        # ids have already convert to label in process_single function
        # palette = palette[labels_ids % 1000]
        palette = palette[labels_ids]
        panoptic_result = draw_masks(img, masks, palette)
        return panoptic_result

    def show_instance(self, img, segms, bboxes, scores, labels, score_thr=0.5):
        if score_thr > 0:
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            segms = segms[inds, ...]
            labels = labels[inds]
        palette = np.asarray(self.PALETTE)
        palette = palette[labels]

        instance_result = draw_masks(img, segms, palette)
        class_name = np.array(self.CLASSES)
        instance_result = imshow_bboxes(
            instance_result, bboxes, class_name[labels], show=False)
        return instance_result

    def show_semantic(self, img, seg_pred, alpha=0.5, palette=None):

        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < alpha <= 1.0
        color_seg = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3),
                             dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg_pred == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - alpha) + color_seg * alpha
        img = img.astype(np.uint8)

        return img


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def draw_masks(img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]

        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha

    p = PatchCollection(
        polygons, facecolor='none', edgecolors='w', linewidths=1, alpha=0.8)

    return img


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole
