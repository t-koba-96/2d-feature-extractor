import torch
import torch.nn as nn

import os
import numpy as np
import argparse
import yaml
from addict import Dict

from make_dataset import make_path_list 
from models.cnn import ImageEncoder , Resnet_BlockEncoder
from utils.dataset import Image_dset


'''
default == using 'cuda:0'
'''

def get_arguments():
    
    parser = argparse.ArgumentParser(description='testing network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('--no_cuda', action = "store_true", help='disable cuda')
    parser.add_argument('--device', type=str, default='0', help='choose device')


    return parser.parse_args()



def main():

     # config
     args = get_arguments()
     SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'), encoding='utf8')))
     print(args)
     if len(args.device) > 1: 
         args.device = list (map(str,args.device))
     os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.device)
     print("using gpu number {}".format(",".join(args.device)))
     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

     # which model
     if SETTING.feature == "block":
        net = Resnet_BlockEncoder(cnn_type = SETTING.model)
     else:
        net = ImageEncoder(cnn_type = SETTING.model)
     net = net.to(device)
     net.eval()

     # dataset
     videos , file_names = make_path_list(SETTING.root, SETTING.video_dir)
     
     # extract
     for vid_num,video in enumerate(videos):
         dset = Image_dset(video , SETTING.image_size)
         print(len(dset))
         dataloader = torch.utils.data.DataLoader(dset,batch_size=SETTING.batch_size,
                                                   shuffle=False,num_workers=SETTING.num_workers)
        
         print("starting video {}".format(file_names[vid_num]))
         with torch.no_grad():
             for i,data in enumerate(dataloader):
                 img, name = data
                 img = img.to(device)
                 output = net(img)

                 if i == 0:
                     tensor = output
                 else:
                     tensor = torch.cat((tensor,output),dim=1)
                 

             print("finished video {}, saving...".format(file_names[vid_num]))
        
             savedir = os.path.join(SETTING.root, SETTING.output_dir, args.arg)
             if not os.path.exists(savedir):
                 os.makedirs(savedir)
             print(tensor.shape)
             np.save(os.path.join(savedir,file_names[vid_num]) , tensor.cpu())

     print("Done!")

if __name__ == '__main__':
    main()