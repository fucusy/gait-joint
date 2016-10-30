__author__ = 'fucus'

import os
import subprocess

import numpy as np
from skimage.io import imsave, imread
import skimage.color as color

import config
from tools import tool, np_helper
import re


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

    mean_filename = "%s/avg.jpg" % back_folder
    if not os.path.exists(mean_filename):
        back_img_path = tool.load_img_path_list(back_folder)
        back_img = tool.img_path_2_pic(back_img_path)
        mean_back_img = np.mean(back_img, axis=0)
        mean_back_img = mean_back_img.astype(np.int32).astype(np.uint8)
        imsave(mean_filename, mean_back_img)
    else:
        mean_back_img = imread(mean_filename)


    img_path = tool.load_img_path_list(img_folder, re.compile(r"\d{4}\.jpg"))
    # do subtraction
    img_data = tool.img_path_2_pic(img_path)
    grey_img_data = np.array([color.rgb2gray(x) for x in img_data])
    back_grey_img = color.rgb2grey(mean_back_img)

    box_filename = "%s/box_file.txt" % img_folder
    box_file = open(box_filename, 'w')
    box_file.write("\t".join(["img", "left-up-height-i", "left-up-width-i", "box-height", "box-width"]))
    box_file.write("\n")

    for i, img in enumerate(grey_img_data):
        base = os.path.basename(img_path[i])

        save_filename = "%s/%s_cover.bmp" % (img_folder, os.path.splitext(base)[0])
        res = tool.subtract(img, back_grey_img, save_filename)

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
    subprocess.call(["cd", config.project.human_pose_path])
    subprocess.call(["th", "test.lua", "-useGPU", "1", "-img_folder", crop_folder])


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

    box_filename = "%s/box_file.txt" % joint_folder
    joints = []
    img_filenames = []
    boxs = []

    is_first = True
    for line in open(box_filename):
        if is_first:
            is_first = False
            continue
        split = line.rstrip('\n').split('\t')
        if len(split) < 5:
            continue
        img_filename = "%s/%s" % (joint_folder, split[0])
        img_base, ext = os.path.splitext(split[0])
        joint_filename = "%s/%s_extract_joint.txt" % (joint_folder, img_base)
        joint = []
        boxs.append([int(x) for x in split[1:]])

        if os.path.exists(joint_filename):
            for j in open(joint_filename):
                split = j.rstrip('\n').split('\t')
                relative_width, relative_height = [int(x) for x in split]
                joint.append([boxs[-1][1]+relative_width, boxs[-1][0]+relative_height])
        joints.append(joint)
        img_filenames.append(img_filename)

    img_data = tool.img_path_2_pic(img_filenames)

    for i in range(len(joints)):
        img_base, ext = os.path.splitext(os.path.basename(img_filenames[i]))
        plot_filename = "%s/%s_plot_joint.jpg" % (joint_folder, img_base)
        plot_img = tool.plot_joint(img_data[i], joints[i])
        plot_box = tool.plot_box(plot_img, boxs[i])
        imsave(plot_filename, plot_box)

    joint_video = "%s/plot_joint.avi" % joint_folder
    img_tpl = "%s/%%04d_plot_joint.jpg" % joint_folder
    subprocess.call(["ffmpeg", "-i", img_tpl, "-c:v", "libx264", joint_video, "-y"])

    return joint_video


def test():
    video_path = config.data.test_video_path
    back_path = config.data.test_back_path
    img_folder = do_box(video_path, back_path)
    try:
        do_joint(img_folder)
    except:
        print("no torch found")
    recover_video_with_joint(img_folder)


if __name__ == '__main__':
    test()