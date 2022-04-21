from easycv.utils.registry import Registry
from easycv.utils.registry import build_from_cfg

OPTIMIZER_BUILDERS = Registry('optimizer builder')
def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)
