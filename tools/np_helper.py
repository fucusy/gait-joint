__author__ = 'fucus'
import numpy as np

def shift_left(img, left=10.0):
    """

    :param numpy.array img: represented by numpy.array
    :param float left: how many pixels to shift to left, this value can be negative that means shift to
                    right {-left} pixels
    :return: numpy.array
    """
    if 0 < abs(left) < 1:
        left = int(left * img.shape[1])
    else:
        left = int(left)

    if len(img.shape) == 1:
        is_grey = True
    else:
        is_grey = False

    img_shift_left = np.zeros(img.shape)
    if left >= 0:
        if is_grey:
            img_shift_left = img[:, left:]
        else:
            img_shift_left = img[:, left:, :]
    else:
        if is_grey:
            img_shift_left = img[:, :left]
        else:
            img_shift_left = img[:, :left, :]

    return img_shift_left


def shift_right(img, right=10.0):
    return shift_left(img, -right)


def shift_up(img, up=10.0):
    """
    :param numpy.array img: represented by numpy.array
    :param float up: how many pixels to shift to up, this value can be negative that means shift to
                    down {-up} pixels
    :return: numpy.array
    """


    if 0 < abs(up) < 1:
        up = int(up * img.shape[0])
    else:
        up = int(up)

    if len(img.shape) == 1:
        is_grey = True
    else:
        is_grey = False

    img_shift_up = np.zeros(img.shape)
    if up >= 0:
        if is_grey:
            img_shift_up = img[up:, :]
        else:
            img_shift_up = img[up:, :, :]
    else:
        if is_grey:
            img_shift_up = img[:up, :]
        else:
            img_shift_up = img[:up, :, :]

    return img_shift_up


def shift_down(img, down=10.0):
    return shift_up(img, -down)


def extract_np(img, left_up_corner, box_size):
    """

    :param img: numpy array, test rbg numpy array
    :param left_up_corner: [{i}, {j}]
    :param box_size:    [{height}, {width}]
    :return: img after crop
    """

    height = img.shape[0]
    width = img.shape[1]

    up_blank = left_up_corner[0]
    left_blank = left_up_corner[1]

    right_blank = width - left_blank - box_size[1]
    down_blank = height - up_blank - box_size[0]


    img = shift_left(img, left_blank)
    img = shift_right(img, right_blank)
    img = shift_up(img, up_blank)
    img = shift_down(img, down_blank)
    return img