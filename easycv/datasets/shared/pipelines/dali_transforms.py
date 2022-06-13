# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np

from easycv.datasets.registry import PIPELINES


@PIPELINES.register_module()
class DaliImageDecoder(object):
    """refer to: https://docs.nvidia.com/deeplearning/dali/archives/dali_0250/user-guide/docs/supported_ops.html#nvidia.dali.ops.ImageDecoder
    """

    def __init__(self, device='mixed', **kwargs):
        import nvidia.dali.ops as ops

        self.decode_op = ops.ImageDecoder(device=device, **kwargs)

    def __call__(self, images):
        return self.decode_op(images)


@PIPELINES.register_module()
class DaliRandomResizedCrop(object):
    """refer to: https://docs.nvidia.com/deeplearning/dali/archives/dali_0250/user-guide/docs/supported_ops.html#nvidia.dali.ops.RandomResizedCrop
    """

    def __init__(self, size, random_area, device='gpu', **kwargs):
        import nvidia.dali.ops as ops

        self.random_resize_crop_op = ops.RandomResizedCrop(
            size=size, random_area=random_area, device=device, **kwargs)

    def __call__(self, images, **kwargs):
        return self.random_resize_crop_op(images, **kwargs)


@PIPELINES.register_module()
class DaliResize(object):
    """refer to: https://docs.nvidia.com/deeplearning/dali/archives/dali_0250/user-guide/docs/supported_ops.html#nvidia.dali.ops.Resize
    """

    def __init__(self, resize_shorter, device='gpu', **kwargs):
        import nvidia.dali.ops as ops
        import nvidia.dali.types as types

        _kwargs = dict(interp_type=types.INTERP_TRIANGULAR)
        _kwargs.update(kwargs)
        self.resize_op = ops.Resize(
            device=device, resize_shorter=resize_shorter, **_kwargs)

    def __call__(self, images, **kwargs):
        return self.resize_op(images, **kwargs)


@PIPELINES.register_module()
class DaliColorTwist(object):
    """refer to: https://docs.nvidia.com/deeplearning/dali/archives/dali_0250/user-guide/docs/supported_ops.html#nvidia.dali.ops.ColorTwist
    """

    def __init__(self,
                 prob,
                 saturation,
                 contrast,
                 brightness,
                 hue,
                 device='gpu',
                 center=1):
        import nvidia.dali.ops as ops
        import nvidia.dali.types as types

        self.color_twist_op = ops.ColorTwist(device=device)
        self.saturation_op = ops.Uniform(
            range=[center - saturation, center + saturation])
        self.contrast_op = ops.Uniform(
            range=[center - contrast, center + contrast])
        self.brightness_op = ops.Uniform(
            range=[center - brightness, center + brightness])
        self.hue_op = ops.Uniform(range=[-hue, hue])

        self.coin_flip_op = ops.CoinFlip(probability=prob)
        self.cast_fp32_op = ops.Cast(dtype=types.FLOAT)

    def __call__(self, images):
        # saturation: default = 1.0 no change
        # brightness: default = 1.0 no change
        # contrast: default = 1.0 no change
        # hue: default = 0.0 no change

        coin_flip = self.cast_fp32_op(self.coin_flip_op())

        saturation = ((1 - coin_flip) * 1.0 + coin_flip * self.saturation_op())
        contrast = ((1 - coin_flip) * 1.0 + coin_flip * self.contrast_op())
        brightness = ((1 - coin_flip) * 1.0 + coin_flip * self.brightness_op())
        hue = coin_flip * self.hue_op()

        output = self.color_twist_op(
            images,
            saturation=saturation,
            contrast=contrast,
            brightness=brightness,
            hue=hue)
        return output


@PIPELINES.register_module()
class DaliRandomGrayscale(object):
    """refer to: https://docs.nvidia.com/deeplearning/dali/archives/dali_0250/user-guide/docs/supported_ops.html#nvidia.dali.ops.Hsv
    Create `RandomGrayscale` op with ops.Hsv.
    when saturation=0, it represents a grayscale image
    """

    def __init__(self, prob, device='gpu'):
        import nvidia.dali.ops as ops
        import nvidia.dali.types as types

        self.coin_flip_op = ops.CoinFlip(probability=1 - prob)
        self.cast_fp32_op = ops.Cast(dtype=types.FLOAT)
        self.hsv_op = ops.Hsv(device=device)

    def __call__(self, images):
        saturation = self.cast_fp32_op(self.coin_flip_op())
        output = self.hsv_op(images, saturation=saturation)

        return output


@PIPELINES.register_module()
class DaliCropMirrorNormalize(object):
    """refer to: https://docs.nvidia.com/deeplearning/dali/archives/dali_0250/user-guide/docs/supported_ops.html#nvidia.dali.ops.CropMirrorNormalize
    """

    def __init__(self,
                 crop,
                 mean,
                 std,
                 prob=0.0,
                 device='gpu',
                 crop_pos_x=0.5,
                 crop_pos_y=0.5,
                 **kwargs):
        import nvidia.dali.ops as ops
        import nvidia.dali.types as types

        _kwargs = dict(dtype=types.FLOAT, output_layout=types.NCHW)
        _kwargs.update(kwargs)

        if isinstance(crop, int):
            crop = [crop, crop]

        # TODO: support random crop_pos by iter
        # We designed crop_pos_x and crop_pos_y random by epoch
        # we has reproduced the accuracy by epoch
        if isinstance(crop_pos_x, (tuple, list)):
            assert len(crop_pos_x) == 2
            crop_pos_x = np.random.uniform(
                low=crop_pos_x[0], high=crop_pos_x[1])
        if isinstance(crop_pos_y, (tuple, list)):
            assert len(crop_pos_y) == 2
            crop_pos_y = np.random.uniform(
                low=crop_pos_y[0], high=crop_pos_y[1])

        self.device = device
        self.coin_flip_op = ops.CoinFlip(probability=prob)
        self.cmn_op = ops.CropMirrorNormalize(
            device=device,
            crop=crop,
            # image_type=image_type,
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y,
            mean=mean,
            std=std,
            **_kwargs)

    def __call__(self, images, **kwargs):
        if self.device == 'gpu':
            images = images.gpu()

        output = self.cmn_op(
            images,
            mirror=self.coin_flip_op(),  # flipped (mirrored) horizontally.
            **kwargs)

        return output
