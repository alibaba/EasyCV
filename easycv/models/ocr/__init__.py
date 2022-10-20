# Copyright (c) Alibaba, Inc. and its affiliates.
from . import backbones
from .cls import TextClassifier
from .det import DBNet
from .heads import CTCHead, DBHead
from .necks import DBFPN, LKPAN, RSEFPN, SequenceEncoder
from .rec import OCRRecNet
