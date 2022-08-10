# Copyright (c) Alibaba, Inc. and its affiliates.
from PIL import ExifTags


def exif_size(img):
    # Get orientation exif tag
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    # Returns exif-corrected PIL size
    s = img.size  # (width, height)

    rotation = dict(img._getexif().items())[orientation]
    if rotation == 6:  # rotation 270
        s = (s[1], s[0])
    elif rotation == 8:  # rotation 90
        s = (s[1], s[0])

    return s
