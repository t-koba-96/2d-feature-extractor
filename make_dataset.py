import torch
from addict import Dict
import yaml
import os
import glob
import argparse
import shutil
import cv2


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('arg', type=str, help='arguments file name')

    return parser.parse_args()


def make_path_list(root , video_path):

    video_list=sorted(glob.glob(os.path.join(root, video_path,"*")))

    file_names = []

    for f in video_list:
        file_names.append(os.path.split(f)[1])

    return video_list , file_names



# Run to make vid2img
if __name__ == '__main__':
    args = get_arguments()
    SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'), encoding='utf8')))
    # vid2img
    video_list = video2image(SETTING.root_path)
    # make path list
    vid, lab, pos, val_vid, val_lab, val_pos= make_path_list(SETTING.root_path,SETTING.train_data)
    print(vid)
    # make class list
    class_list = make_class_list()
    print(class_list)


    