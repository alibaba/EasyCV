# Copyright (c) Alibaba, Inc. and its affiliates.
from .african_wildlife import DetSourceAfricanWildlife
from .coco import DetSourceCoco, DetSourceCoco2017
from .coco_livs import DetSourceLvis
from .coco_panoptic import DetSourceCocoPanoptic
from .fruit import DetSourceFruit
from .pai_format import DetSourcePAI
from .pet import DetSourcePet
from .raw import DetSourceRaw
from .voc import DetSourceVOC, DetSourceVOC2007, DetSourceVOC2012
from .wider_person import DetSourceWiderPerson

__all__ = [
    'DetSourceCoco', 'DetSourceCocoPanoptic', 'DetSourcePAI', 'DetSourceRaw',
    'DetSourceVOC', 'DetSourceVOC2007', 'DetSourceVOC2012',
    'DetSourceCoco2017', 'DetSourceLvis', 'DetSourceWiderPerson',
    'DetSourceAfricanWildlife', 'DetSourcePet'
]
