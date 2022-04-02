# Copyright (c) Alibaba, Inc. and its affiliates.
import os


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include)
                and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def get_parent_path(path: str):
    """get parent path, support oss-style path
    """
    eles = path.rstrip(os.sep).split(os.sep)
    parent = os.sep.join(eles[:-1]) + os.sep
    return parent
