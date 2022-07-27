# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.utils.torchacc_util import is_torchacc_enabled

if is_torchacc_enabled():
    import torchacc.torch_xla.distributed.parallel_loader as pl

    class TorchaccLoaderWrapper(pl.MpDeviceLoader):

        def __init__(self, loader, device=None, **kwargs) -> None:
            if device is None:
                import torchacc.torch_xla.core.xla_model as xm
                device = xm.xla_device()

            super(TorchaccLoaderWrapper, self).__init__(
                loader=loader, device=device, **kwargs)

        @property
        def sampler(self):
            return self._loader.sampler
else:
    TorchaccLoaderWrapper = None
