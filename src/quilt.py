import numpy as np
from utils.enum import Enum
import io
import tile


Tiling = Enum(["SIMPLE", "MATCHED"])


def synthesize(img_path, block_size, tiling=None, magnify_by=2, overlap_size=None):
    if img_path is None:
        return

    if block_size is None:
        return

    if tiling is None:
        tiling = Tiling.MATCHED

    image = io.get_img(img_path)
    block_size = np.asarray(block_size)

    if tiling == Tiling.SIMPLE:
        tile.simple_tiling(image, block_size, magnify_by)
    else:
        if overlap_size is None:
            overlap_size = block_size / 6

        overlap_size = np.asarray(overlap_size)


synthesize("images\\texture.jpg", [10, 10])
