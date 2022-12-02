# Copyright (c) Alibaba, Inc. and its affiliates.
from .african_wildlife import DetSourceAfricanWildlife
from .artaxor import DetSourceArtaxor
from .coco import DetSourceCoco, DetSourceCoco2017, DetSourceTinyPerson
from .coco_livs import DetSourceLvis
from .coco_panoptic import DetSourceCocoPanoptic
from .crowd_human import DetSourceCrowdHuman
from .fruit import DetSourceFruit
from .objects365 import DetSourceObjects365
from .pai_format import DetSourcePAI
from .pet import DetSourcePet
from .raw import DetSourceRaw
from .voc import DetSourceVOC, DetSourceVOC2007, DetSourceVOC2012
from .wider_face import DetSourceWiderFace
from .wider_person import DetSourceWiderPerson

__all__ = [
    'DetSourceCoco', 'DetSourceCocoPanoptic', 'DetSourceObjects365',
    'DetSourcePAI', 'DetSourceRaw', 'DetSourceVOC', 'DetSourceVOC2007',
    'DetSourceVOC2012', 'DetSourceCoco2017'
    'DetSourceCoco', 'DetSourceCocoPanoptic', 'DetSourcePAI', 'DetSourceRaw',
    'DetSourceVOC', 'DetSourceVOC2007', 'DetSourceVOC2012',
    'DetSourceCoco2017', 'DetSourceLvis', 'DetSourceWiderPerson',
    'DetSourceAfricanWildlife', 'DetSourcePet', 'DetSourceWiderFace',
    'DetSourceCrowdHuman'
]
