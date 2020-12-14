import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


class Image_dataset(object):
    def __init__(self, video_root, image_size = 224):

        self.root = video_root
        self.imgs = list(sorted(os.listdir(video_root)))
        self.spatial_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])
  
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img_name = img_path
        img = self.spatial_transform(img)

        return img , img_name
        
    def __len__(self):
        return len(self.imgs)
