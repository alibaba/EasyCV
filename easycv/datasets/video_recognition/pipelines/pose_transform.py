# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) OpenMMLab. All rights reserved.
# Refer to: https://github.com/open-mmlab/mmaction2/blob/master/mmaction/datasets/pipelines/pose_loading.py
import numpy as np

from easycv.datasets.registry import PIPELINES


@PIPELINES.register_module()
class PaddingWithLoop:
    """Sample frames from the video.

    To sample an n-frame clip from the video, PaddingWithLoop samples
    the frames from zero index, and loop the frames if the length of
    video frames is less than te value of 'clip_len'.

    Required keys are "total_frames", added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
    """

    def __init__(self, clip_len, num_clips=1):

        self.clip_len = clip_len
        self.num_clips = num_clips

    def __call__(self, results):
        num_frames = results['total_frames']

        start = 0
        inds = np.arange(start, start + self.clip_len)
        inds = np.mod(inds, num_frames)

        results['frame_inds'] = inds.astype(np.int64)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score"
    (optional), added or modified keys are "keypoint", "keypoint_score" (if
    applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        """Load keypoints given frame indices.

        Args:
            kp (np.ndarray): The keypoint coordinates.
            frame_inds (np.ndarray): The frame indices.
        """

        return [x[frame_inds].astype(np.float32) for x in kp]

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        """Load keypoint scores given frame indices.

        Args:
            kpscore (np.ndarray): The confidence scores of keypoints.
            frame_inds (np.ndarray): The frame indices.
        """

        return [x[frame_inds].astype(np.float32) for x in kpscore]

    def __call__(self, results):

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            kpscore = results['keypoint_score']
            results['keypoint_score'] = kpscore[:,
                                                frame_inds].astype(np.float32)

        if 'keypoint' in results:
            results['keypoint'] = results['keypoint'][:, frame_inds].astype(
                np.float32)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class PoseNormalize:
    """Normalize the range of keypoint values to [-1,1].

    Args:
        mean (list | tuple): The mean value of the keypoint values.
        min_value (list | tuple): The minimum value of the keypoint values.
        max_value (list | tuple): The maximum value of the keypoint values.
    """

    def __init__(self,
                 mean=(960., 540., 0.5),
                 min_value=(0., 0., 0.),
                 max_value=(1920, 1080, 1.)):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1, 1)
        self.min_value = np.array(
            min_value, dtype=np.float32).reshape(-1, 1, 1, 1)
        self.max_value = np.array(
            max_value, dtype=np.float32).reshape(-1, 1, 1, 1)

    def __call__(self, results):
        keypoint = results['keypoint']
        keypoint = (keypoint - self.mean) / (self.max_value - self.min_value)
        results['keypoint'] = keypoint
        results['keypoint_norm_cfg'] = dict(
            mean=self.mean, min_value=self.min_value, max_value=self.max_value)
        return results


@PIPELINES.register_module()
class FormatGCNInput:
    """Format final skeleton shape to the given input_format.

    Required keys are "keypoint" and "keypoint_score"(optional),
    added or modified keys are "keypoint" and "input_shape".

    Args:
        input_format (str): Define the final skeleton format.
    """

    def __init__(self, input_format, num_person=2):
        self.input_format = input_format
        if self.input_format not in ['NCTVM']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')
        self.num_person = num_person

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']

        if 'keypoint_score' in results:
            keypoint_confidence = results['keypoint_score']
            keypoint_confidence = np.expand_dims(keypoint_confidence, -1)
            keypoint_3d = np.concatenate((keypoint, keypoint_confidence),
                                         axis=-1)
        else:
            keypoint_3d = keypoint

        keypoint_3d = np.transpose(keypoint_3d,
                                   (3, 1, 2, 0))  # M T V C -> C T V M

        if keypoint_3d.shape[-1] < self.num_person:
            pad_dim = self.num_person - keypoint_3d.shape[-1]
            pad = np.zeros(
                keypoint_3d.shape[:-1] + (pad_dim, ), dtype=keypoint_3d.dtype)
            keypoint_3d = np.concatenate((keypoint_3d, pad), axis=-1)
        elif keypoint_3d.shape[-1] > self.num_person:
            keypoint_3d = keypoint_3d[:, :, :, :self.num_person]

        results['keypoint'] = keypoint_3d
        results['input_shape'] = keypoint_3d.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str
