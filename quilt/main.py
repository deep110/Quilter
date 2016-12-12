import numpy as np
from utils.enum import Enum
import io
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
        tiling = Tiling.SIMPLE

    img_path = base_path + "/" + img_path
    image = io.get_img(img_path)
    block_size = np.asarray(block_size)

    if tiling == Tiling.SIMPLE:
        result = tile.simple_tiling(image, block_size, magnify_by)
    else:
        if overlap_size is None:
            overlap_size = block_size / 6

        result = np.asarray(overlap_size)

    return result
