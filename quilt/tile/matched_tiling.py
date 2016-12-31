import numpy as np


def matched_tiling(img, block_size, target_shape, overlap_size):

    new_block_size = block_size - overlap_size
    n_blocks = (np.ceil(np.true_divide(target_shape[0:2], new_block_size))).astype('uint32')

    output = np.zeros(target_shape, 'uint8')
    print("Total blocks to build: " + str(n_blocks[0] * n_blocks[1]))
    print("Building...")

    for i in range(0, target_shape[0]):
        row1 = i * new_block_size[0]
        row2 = min((i + 1) * new_block_size[0] + overlap_size[0], target_shape[1])

        if row2 - row1 < overlap_size[0]:
            continue

        for j in range(0, target_shape[1]):
            col1 = j * new_block_size[1]
            col2 = min((j + 1) * new_block_size[1] + overlap_size[1], target_shape[1])

            if col2 - col1 < overlap_size[1]:
                continue

            total_n = i * n_blocks[1] + j + 1
            print('Building block ' + str(total_n))

            anchor = Anchor(output[row1:row2, col1:col2], row1, col1, overlap_size)
            patch = matched_crop(img, np.asarray([row2 - row1, col2 - col1]), anchor)
            anchor.stitch(patch)

    return output


class Anchor(object):

    def __init__(self, data, row_index, col_index, overlap_size):
        self.__data = data
        self.__row_index = row_index
        self.__col_index = col_index
        self.__overlap_size = overlap_size

    def __calc_pixel_error(self, block1, block2):
        diff = block1 - block2
        r, g, b = self.__split_channels(diff * diff)
        r *= 0.30
        g *= 0.59
        b *= 0.11

        return np.sqrt(r + g + b)

    def calc_error(self, patch):
        if self.__row_index == 0 and self.__col_index == 0:
            return 0.0

        total_error = 0.0

        # calculate vertical strip error (includes overlap region)
        if self.__col_index != 0:
            b1 = self.__data[:, : self.__overlap_size[1]]
            b2 = patch[:, : self.__overlap_size[1]]
            total_error += np.sum(self.__calc_pixel_error(b1, b2))

        # calculate horizontal strip error (excludes overlap region, since its 
        # already included above)
        if self.__row_index != 0:
            b1 = self.__data[: self.__overlap_size[0], self.__overlap_size[1]:]
            b2 = patch[: self.__overlap_size[0], self.__overlap_size[1]:]
            total_error += np.sum(self.__calc_pixel_error(b1, b2))

        return total_error

    def stitch(self, patch):
        if self.__row_index == 0 and self.__col_index == 0:
            return patch

    @staticmethod
    def __split_channels(array):
        r = array[:, :, 0]
        g = array[:, :, 1]
        b = array[:, :, 2]

        return r, g, b


def matched_crop(img, block_size, anchor):
    """
    img src image to crop
    size size of rect to cut [m,n]
    anchor target to match to
    """
    img_size = (np.asarray(img.shape))[0:2]
    max_size = img_size - block_size
    error = np.ones(max_size)
    min_error = np.inf

    for i in range(0, max_size[0]):
        for j in range(0, max_size[1]):
            patch = img[i:i + block_size[0], j:j + block_size[1]]
            curr_error = anchor.calc_error(patch)
            error[i, j] = curr_error

            if curr_error < min_error:
                min_error = curr_error

            if min_error == 0.0:
                break

    threshold = min_error * 1.1
    mask = (error <= threshold).nonzero()
    possible = len(mask[0])
    to_take = np.random.randint(0, possible)

    row_index = (mask[0])[to_take]
    col_index = (mask[1])[to_take]

    return img[row_index:row_index + block_size[0], col_index:col_index + block_size[1]]
