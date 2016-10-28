__author__ = 'fucus'
import os
import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import skimage.color as color
from skimage import filter
from skimage.morphology import disk
import config

def extract_info_from_path(path):
    """

    :param path: xxxx/xxx/xxx/{hid}-{cond}-{seq}-{view}-xxx.xxx
        for example: /Users/fucus/Documents/irip/gait_recoginition/code/001-nm-01-090.avi
    :return: {hid}, {cond}, {seq}, {view}
    """
    img_id = ''.join(os.path.basename(path).split('.')[:-1])
    split_img_id = img_id.split('-')
    hid = split_img_id[0]
    cond = split_img_id[1]
    seq = split_img_id[2]
    view = split_img_id[3]
    return hid, cond, seq, view


def load_img_path_list(path):
    """

    :param path: the test img folder
    :return:
    """
    list_path = os.listdir(path)
    # change to reg to match extension
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg") or x.endswith("png") or x.endswith("bmp")]
    return np.array(result)


def img_path_2_pic(img_paths, func=None):
    img_pics = []
    for img_path in img_paths:
        im = imread(img_path)
        if func is not None:
            im = func(im)
        img_pics.append(im)
    return np.array(img_pics)


def im_show(im):
    """

    :param im: numpy.array
    :return:
    """
    plt.imshow(im)
    plt.draw()


def get_human_position(img):
    """
    :param img: grey type numpy.array image
    :return: left up corner and width, height of the box,
        [{left-up-corner-i}, {left-up-corner-j}, {box-i}, {box-j}]

        you can get the left up pixel by img[{left-up-corner-i}][{left-up-corner-j}]
        right bottom pixel by img[{left-up-corner-i}+{box-i}][{left-up-corner-j}+{box-j}]
    """

    left_blank = 0
    right_blank = 0

    up_blank = 0
    down_blank = 0

    height = img.shape[0]
    width = img.shape[1]

    for i in range(height):
        if np.sum(img[i, :]) == 0:
            up_blank += 1
        else:
            break

    for i in range(height-1, -1, -1):
        if np.sum(img[i, :]) == 0:
            down_blank += 1
        else:
            break

    for i in range(width):
        if np.sum(img[:, i]) == 0:
            left_blank += 1
        else:
            break

    for i in range(width-1, -1, -1):
        if np.sum(img[:, i]) == 0:
            right_blank += 1
        else:
            break

    left_up_corner_i = up_blank
    left_up_corner_j = left_blank
    box_i = max(0, height - up_blank - down_blank)
    box_j = max(0, width - left_blank - right_blank)
    return [left_up_corner_i, left_up_corner_j, box_i, box_j]


def subtract(img, back):
    """

    :param img: numpy array
    :param back: numpy array
    :return: a black and white img stored in numpy array
    """

    if len(img.shape) == 3 and img.shape[2] == 3:
        img = color.rgb2grey(img)

    if len(back.shape) == 3 and back.shape[2] == 3:
        back = color.rgb2grey(back)

    sub_img = np.subtract(img, back)
    sub_img = np.absolute(sub_img)
    clap = 0.08
    low_values_indices = sub_img < clap  # Where values are low
    sub_img[low_values_indices] = 0  # All low values set to 0


    sub_img_uint8 = sub_img * 255
    sub_img_uint8 = sub_img_uint8.astype(np.uint8)
    sub_img_uint8 = filter.median(sub_img_uint8, disk(5))


    # sub_img = sub_img_uint8.astype(np.float32) / 255
    # count = 0
    # filename = "%s/test_img/sub_img%5d.bmp" % (config.project.data_path, count)
    # while os.path.exists(filename):
    #     count += 1
    #     filename = "%s/test_img/sub_img%05d.bmp" % (config.project.data_path, count)
    #
    # dirname = os.path.dirname(filename)
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    # imsave(filename, sub_img)

    return get_human_position(sub_img_uint8)