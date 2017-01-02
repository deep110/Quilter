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
            output[row1:row2, col1:col2] = anchor.stitch(patch)

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
            b1 = self.__data[:, 0: self.__overlap_size[1]]
            b2 = patch[:, 0: self.__overlap_size[1]]
            total_error += np.sum(self.__calc_pixel_error(b1, b2))

        # calculate horizontal strip error (excludes overlap region, since its 
        # already included above)
        if self.__row_index != 0:
            b1 = self.__data[0: self.__overlap_size[0], self.__overlap_size[1]:]
            b2 = patch[: self.__overlap_size[0], self.__overlap_size[1]:]
            total_error += np.sum(self.__calc_pixel_error(b1, b2))

        return total_error

    def stitch(self, patch):
        if self.__row_index == 0 and self.__col_index == 0:
            return patch

        vertical_error, vertical_offset = None, None
        horizontal_error, horizontal_offset = None, None

        if self.__col_index != 0:
            b1 = self.__data[:, 0: self.__overlap_size[1]]
            b2 = patch[:, 0: self.__overlap_size[1]]

            vertical_error = self.__calc_pixel_error(b1, b2)
            error_shape = vertical_error.shape
            vertical_offset = np.zeros(error_shape, np.int8)

            for i in range(error_shape[0]-2, -1, -1):
                for j in range(0, error_shape[1]):
                    fix_offset = -1
                    s_i = j-1
                    if s_i < 0:
                        s_i = 0
                        fix_offset = 0

                    e_i = j+2
                    if e_i > error_shape[1]:
                        e_i = error_shape[1]
                    temp = vertical_error[i-1, s_i:e_i]
                    vertical_error[i, j] += np.min(temp)
                    vertical_offset[i, j] = np.argmin(temp)+fix_offset

        if self.__row_index != 0:
            b1 = self.__data[: self.__overlap_size[0], :]
            b2 = patch[: self.__overlap_size[0], :]

            horizontal_error = self.__calc_pixel_error(b1, b2)
            error_shape = horizontal_error.shape
            horizontal_offset = np.zeros(error_shape, np.int8)
            for i in range(error_shape[1] - 2, -1, -1):
                for j in range(0, error_shape[0]):
                    fix_offset = -1
                    s_i = j - 1
                    if s_i < 0:
                        s_i = 0
                        fix_offset = 0

                    e_i = j + 2
                    if e_i > error_shape[1]:
                        e_i = error_shape[1]
                    temp = horizontal_error[s_i:e_i, i-1]
                    horizontal_error[j, i] += np.min(temp)
                    horizontal_offset[j, i] = np.argmin(temp) + fix_offset

        if self.__row_index == 0:
            min_j = np.argmin(vertical_error[0, :])
            patch[0, :min_j] = self.__data[0, :min_j]
            for i in range(1, vertical_error.shape[0]):
                min_j += vertical_offset[i-1, min_j]
                patch[i, 0:min_j] = self.__data[i, 0:min_j]

        elif self.__col_index == 0:
            min_i = np.argmin(horizontal_error[:, 0])
            patch[0:min_i, 0] = self.__data[0:min_i, 0]
            for j in range(1, horizontal_error.shape[1]):
                min_i += horizontal_offset[min_i, j-1]
                patch[0:min_i, j] = self.__data[0:min_i, j]

        else:
            error = vertical_error[0:self.__overlap_size[0], :] + horizontal_error[:, 0: self.__overlap_size[1]]
            temp = np.argmin(error)
            minI = min_i = temp/error.shape[0]
            minJ = min_j = temp % error.shape[1]

            for j in range(minJ, horizontal_error.shape[1]):
                min_i += horizontal_offset[min_i, j-1]
                patch[0:min_i, j] = self.__data[0:min_i, j]

            for i in range(minI, vertical_error.shape[0]):
                min_j += vertical_offset[i-1, min_j]
                patch[i, 0:min_j] = self.__data[i, 0:min_j]

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
