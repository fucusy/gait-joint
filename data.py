__author__ = 'fucus'

import os
import subprocess

import numpy as np
from skimage.io import imsave
import skimage.color as color

import config
from tools import tool, np_helper



def load_test_video():
    """

    :return: video path list
    """

    res = []
    res.append()
    return ""


def do_box(video_path, back_video_path):
    """
    using ffmpeg to split video to a sequence images at folder named with the video file name
    then get human box data by subtraction and save box data in txt file named image name

    and
    crop every image in img_folder by the human box data, save the crop image at the same
    folder, named specially

    :param video_path:
    :param back_video_path:
    :return: image folder name
    """
    video_name = os.path.basename(video_path)
    hid, cond, seq, view = tool.extract_info_from_path(video_name)

    img_folder = ''.join(video_name.split('.')[:-1])
    img_folder = "%s/%s/%s" % (config.project.data_path, hid, img_folder)

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

        output = "%s/%%04d.jpg" % img_folder
        subprocess.call(["ffmpeg", "-i", video_path, output])

    back_name = os.path.basename(back_video_path)
    back_folder = ''.join(back_name.split('.')[:-1])
    back_folder = "%s/%s/%s" % (config.project.data_path, hid, back_folder)

    if not os.path.exists(back_folder):
        os.makedirs(back_folder)
        output = "%s/%%04d.jpg" % back_folder
        subprocess.call(["ffmpeg", "-i", back_video_path, output])

    # get mean back img
    back_img_path = tool.load_img_path_list(back_folder)
    back_img = tool.img_path_2_pic(back_img_path)
    mean_back_img = np.mean(back_img, axis=0)
    mean_back_img = mean_back_img.astype(np.int32).astype(np.uint8)

    mean_filename = "%s/avg.jpg" % back_folder
    imsave(mean_filename, mean_back_img)

    img_path = tool.load_img_path_list(img_folder)
    # do subtraction
    img_data = tool.img_path_2_pic(img_path)
    grey_img_data = np.array([color.rgb2gray(x) for x in img_data])
    back_grey_img = color.rgb2grey(mean_back_img)

    box_filename = "%s/box_file.txt" % img_folder
    box_file = open(box_filename, 'w')
    box_file.write("\t".join(["img", "left-up-height-i", "left-up-width-i", "box-height", "box-width"]))
    box_file.write("\n")

    for i, img in enumerate(grey_img_data):
        res = tool.subtract(img, back_grey_img)
        base = os.path.basename(img_path[i])
        line = [base]
        for el in res:
            line.append(str(el))
        box_file.write("\t".join(line))
        box_file.write("\n")

        # do crop
        extract_img = np_helper.extract_np(img_data[i], res[0:2], res[2:4])
        extract_img_filename = "%s/%s_extract.jpg" % (img_folder, base.rstrip(".jpg"))
        if len(extract_img) != 0:
            imsave(extract_img_filename, extract_img)
    box_file.close()
    return img_folder


def do_joint(crop_folder):
    """
    save joint data into the same folder named specially
    :param crop_folder:  the folder contain cropped image
    :return:
    """

    return True


def update_joint(joint_folder):
    """
    update joint data by some method, save joint data into special name like v1, v2, v3

    :param joint_folder: the folder contain joint data
    :return:
    """

    return True


def recover_video_with_joint(joint_folder):
    """
    form a video contain the joint data, then save it
    :param joint_folder: the folder contain joint data
    :return: video path
    """
    joint_video = ""
    return joint_video


def test():
    video_path = config.data.test_video_path
    back_path = config.data.test_back_path
    do_box(video_path, back_path)


if __name__ == '__main__':
    test()