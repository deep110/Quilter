from PIL import Image
import numpy as np


def get_img(path):
    return np.asarray(Image.open(path))


def save_img(name, image):
    im = Image.fromarray(np.uint8(image))
    im.save(name, "jpeg")


def show_img(img):
    im = Image.fromarray(np.uint8(img))
    im.show()