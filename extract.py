import numpy as np
import argparse
import shutil
import os
import torch
import yaml
from addict import Dict
import pandas as pd
from tqdm import tqdm

from main.setting import get_videos 
from main.dataset import Image_dataset
from main.models.cnn import ResNet, VGG


'''
default == using 'cuda:0'
'''

def get_arguments():
    
    parser = argparse.ArgumentParser(description='2d feature extractor')
    parser.add_argument('dataset_dir', type=str, help='path to dataset directory')
    parser.add_argument('save_dir', type=str, help='path to the directory you want to save video features')
    parser.add_argument('model', type=str, help='model architecture. [ resnet50 | resnet101 | vgg16 ]')
    parser.add_argument('--spatial_feature', action='store_true', help='if you want to keep the extracted feature in 2d shape, store_true')
    parser.add_argument('--cuda', type=str, default= [0], help='choose cuda num')

    return parser.parse_args()



def main():

    # config
    args = get_arguments()
    print(args)
    args.cuda = list(map(str,args.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    if args.model.startswith('resnet'):
        model = Resnet(args.model, spatial_feature=args.spatial_feature)
    elif args.model.startswith('vgg'):
        model = VGG(args.model, spatial_feature=args.spatial_feature)
    model = model.to(device)
    model.eval()

    # dataset
    vid_paths, vid_names, vid_len = get_videos(args.dataset_dir)

    # make save directory
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    
    # progress bar
    vid_bar = tqdm(total = vid_len)
    vid_bar.set_description('Total Progress Rate')
     
    # for each video
    tensor_list = []
    for i, (vid_path, vid_name) in enumerate(zip(vid_paths, vid_names)):
        # dataloader
        video_set = Image_dataset(vid_path , image_size=224)
        video_loader = torch.utils.data.DataLoader(video_set,batch_size=64,
                                                 shuffle=False, num_workers=4)
        # progress bar
        frame_bar = tqdm(total = video_set.__len__(), leave=False)
        frame_bar.set_description('Video Progress Rate')

        # for each frame
        with torch.no_grad():
            for j, (img, img_path) in enumerate(video_loader):

                img = img.to(device)
                output = model(img)

                if j == 0:
                    tensor = output
                else:
                    tensor = torch.cat((tensor,output),dim=0)

                frame_bar.update(len(img_path))                 


        tensor_list.append(tensor.size())
        np.save(os.path.join(args.save_dir, vid_name), tensor.cpu())
        vid_bar.update(1)
        frame_bar.close()

    # save to result csv
    df = pd.DataFrame({"video": vid_names, "feature shape": tensor_list})
    df.to_csv(os.path.join('result','{}_{}.csv'.format(os.path.basename(args.dataset_dir), args.model)), index=None)

    print("Done!")

if __name__ == '__main__':
    main()