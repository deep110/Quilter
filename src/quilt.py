import numpy as np
import core.io


def synthesize(img_path, block_size, tiling = None):

	i = core.io.get_img(img_path)
	# s = qt.simple_tiling(i, [10,10])
	# o = qt.matched_tiling(i, [100,100], magnify_by = 2)
	core.io.show_img(i)

synthesize("images\\texture.jpg", [10,10])






