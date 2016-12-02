import numpy as np

def simple_tiling(img, block_size, magnify_by = 2):
	block_size = np.asarray(block_size)
	target_img_shape = (magnify_by * np.asarray(img.shape)).astype('uint32')
	target_img_shape[2] = img.shape[2]

	n_blocks = (np.ceil(np.true_divide(target_img_shape[0:2],block_size))).astype('uint32')
	output = np.zeros(target_img_shape,'uint8')

	for i in range(0, n_blocks[0]):
		row1 = i*block_size[0]
		row2 = min((i+1)*block_size[0], target_img_shape[0])

		for j in range(0, n_blocks[1]):
			col1 = j*block_size[1]
			col2 = min((j+1)*block_size[1], target_img_shape[1])

			temp = random_crop(img, block_size)
			output[row1:row2,col1:col2] = temp[0:row2-row1, 0:col2-col1]
	
	return output

"""
img: image to crop
block_size: size of rect to cut [m,n]
"""
def random_crop(img, block_size):
	img_size = (np.asarray(img.shape))[0:2]
	startPix = (np.rint(np.random.rand(2,) * (img_size - block_size))).astype('uint32')
	endPix = startPix + block_size

	return img[startPix[0]:endPix[0], startPix[1]:endPix[1]]
	