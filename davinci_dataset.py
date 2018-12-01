# -*- coding: utf-8 -*-

import os
import glob
import torch
import numpy as np
import torchvision.transforms.functional as tfn
from torch.utils.data.dataset import Dataset
from PIL import Image

class DavinciDataset(Dataset):
    def __init__(self, rootDir, train ,leftmap, rightmap, transform = None, reshapeSize = (192, 96)):
        self.transforms = transform
        self.rootDir = rootDir
        self.leftmap = leftmap
        self.right  = rightmap
        self.reshapeSize = reshapeSize
        
        testrainfolder = 'train' if train else 'test'
        
        path_left = os.path.join(self.rootDir, testrainfolder, leftmap)
        path_right = os.path.join(self.rootDir, testrainfolder, rightmap)
        im_0 = glob.glob(path_left + "/*.png")
        
        self.roots = []

        for i in im_0:
            head, tail = os.path.split(i)
            right_pair = path_right +"/" + tail 
            if os.path.isfile(right_pair):
                self.roots.append((i, right_pair))
                
    def __len__(self):
        return len(self.roots)
    
    def __getitem__(self, idx):
        image_l = Image.open(self.roots[idx][0]).resize(self.reshapeSize, Image.ANTIALIAS)
        image_r = Image.open(self.roots[idx][1]).resize(self.reshapeSize, Image.ANTIALIAS)        
        if self.transforms:
            image_l = self.transforms(image_l)
            image_r = self.transforms(image_r)
        else:
           image_l = tfn.to_tensor(image_l)
           image_r = tfn.to_tensor(image_r)
            
        return (image_l, image_r)

