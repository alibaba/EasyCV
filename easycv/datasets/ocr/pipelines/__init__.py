# Copyright (c) Alibaba, Inc. and its affiliates.
from .det_transform import (EastRandomCropData, IaaAugment, MakeBorderMap,
                            MakeShrinkMap, OCRDetResize)
from .label_ops import CTCLabelEncode, MultiLabelEncode, SARLabelEncode
from .rec_transform import ClsResizeImg, RecAug, RecConAug, RecResizeImg
