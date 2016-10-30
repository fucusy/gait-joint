__author__ = 'fucus'
import os
import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import skimage.color as color
from skimage import filter
from skimage.morphology import disk
from PIL import Image, ImageDraw
import config
import re


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


def load_img_path_list(path, pattern=None):
    """

    :param path: the test img folder
    :return:
    """
    list_path = os.listdir(path)
    filtered = []
    # change to reg to match extension

    p = re.compile(r".*\.[jpg|png|bmp]")
    for x in list_path:
        if re.match(p, x):
            if pattern is not None and re.match(pattern, x):
                filtered.append(x)
            elif pattern is None:
                filtered.append(x)
    result = ["%s/%s" % (path, x) for x in filtered]
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
    plt.clf()
    plt.imshow(im)
    plt.show()
    plt.close()



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


def subtract(img, back, save_filename):
    """

    :param img: numpy array
    :param back: numpy array
    :param save_filename: save the sub result to this filename
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

    sub_img = sub_img_uint8.astype(np.float32) / 255

    dirname = os.path.dirname(save_filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    imsave(save_filename, sub_img)

    return get_human_position(sub_img_uint8)


def plot_joint(img, joints):
    """
    :param img:
    :param joints: joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee,
                    5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top,
                    10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder,
                    14 - l elbow, 15 - l wrist)
    :return: the img after put skeleton on it
    """

    pairs = np.array(
        [[1, 2], [2, 3], [3, 7], [4, 5], [4, 7], [5, 6], [7, 9], [9, 10], [14, 9], [11, 12], [12, 13], [13, 9],
         [14, 15], [15, 16]]) - 1

    colors = [
        (255	,192	,203) #pink
        ,(216	,191	,216) #thistle
        ,(202	,225	,255) #lightsteelblue 1
        ,(152	,245	,255) #cadetblue 2
        ,(84	,255	,159) #sea green 1
        ,(255	,246	,143) #khaki 1
        ,(205	,201	,201) #snow 3
    ]


    img = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    for i in range(pairs.shape[0]):

        if len(joints) > pairs[i, 0] and len(joints) > pairs[i, 1]:
            # plot only the visible joints
            a_x, a_y = joints[pairs[i, 0]]
            b_x, b_y = joints[pairs[i, 1]]
            draw.line((a_x, a_y, b_x, b_y), width=3, fill=colors[i%len(colors)])
    return np.asarray(img)

def plot_box(img, box):
    """

    :param img:
    :param box:left up corner and width, height of the box,
        [{left-up-corner-i}, {left-up-corner-j}, {box-i}, {box-j}]

        you can get the left up pixel by img[{left-up-corner-i}][{left-up-corner-j}]
        right bottom pixel by img[{left-up-corner-i}+{box-i}][{left-up-corner-j}+{box-j}]
    :return:
    """
    img = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    draw.line((box[1], box[0], box[1] + box[3], box[0]))

    draw.line((box[1], box[0]+box[2], box[1] + box[3], box[0] + box[2]))

    draw.line((box[1], box[0], box[1], box[0] + box[2]))

    draw.line((box[1] + box[3], box[0], box[1] + box[3], box[0] + box[2]))

    return np.asarray(img)


def read_list_from(filename):
    joint = []
    for j in open(filename):
        relative_x, relative_y = map(int, j.rstrip('\n').split('\t'))
        joint.append([relative_x, relative_y])
    return joint

if __name__ == '__main__':
    filename = "/Users/fucus/Documents/irip/gait_recoginition/code/gait-joint/data/001/001-nm-01-090/0069_extract.jpg"
    img = imread(filename)
    joint_filename = "/Users/fucus/Documents/irip/gait_recoginition/code/gait-joint/data/001/001-nm-01-090/0069_extract_joint.txt"
    joints = read_list_from(joint_filename)
    plot_img = plot_joint(img, joints)
    im_show(plot_img)

    filename = "/Users/fucus/Documents/irip/gait_recoginition/code/gait-joint/data/001/001-nm-01-090/0069.jpg"
    box = [48, 165, 147, 35]
    img = imread(filename)
    box_img = plot_box(img, box)
    im_show(box_img)