# Copyright (c) Alibaba, Inc. and its affiliates.
import os


def is_torchacc_enabled():
    return bool(int(os.getenv('USE_TORCHACC', '0')))
