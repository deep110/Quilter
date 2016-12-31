import numpy as np
from utils.enum import Enum
import image_io as io
import tile
import os
import sys


Tiling = Enum(["SIMPLE", "MATCHED"])


def synthesize(img_path, block_size, tiling=None, magnify_by=2, overlap_size=None):
    base_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    if img_path is None:
        return

    if block_size is None:
        return

    if tiling is None:
        tiling = Tiling.MATCHED

    img_path = base_path + "/" + img_path
    image = io.get_img(img_path)
    block_size = np.asarray(block_size)

    image_shape = np.asarray(image.shape)
    target_shape = (magnify_by * image_shape)
    if int(image_shape.shape[0]) > 2:
        target_shape[2] = image_shape[2]

    if tiling == Tiling.SIMPLE:
        result = tile.simple_tiling(image, block_size, target_shape)
    else:
        if overlap_size is None:
            overlap_size = block_size / 6

        overlap_size = np.asarray(overlap_size)
        result = tile.matched_tiling(image, block_size, target_shape, overlap_size)

    return result
