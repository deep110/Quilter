import cv2 as cv

def get_img(path):
	img = cv.imread(path,1)
	return img

def save_img(name,image):
	cv.imwrite(name, image)

def show_img(img):
	cv.imshow('image',img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def split_channels(img):
	b,g,r = cv.split(img)
	return r,g,b