__author__ = 'fucus'
import os


class project:
    base_folder = "/Users/fucus/Documents/irip/gait_recoginition/code/gait-joint/"
    data_path = "%s/data" % base_folder
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    human_pose_path = "/home/chenqiang/github/human-pose-estimation"
class data:
    test_video_path = "/Users/fucus/Documents/irip/gait_recoginition/data/001-nm-01-090.avi"
    test_back_path = "/Users/fucus/Documents/irip/gait_recoginition/data/001-bkgrd-090.avi"
    dataset_b_video = "/home/chenqiang/data/CASIA_full_gait_data/DatasetB/videos"
