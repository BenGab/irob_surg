import os
import glob
import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np

class ToolDataset(Dataset):
    def __init__(self, rootDir, mean, std, pixelDiv, input_size):
        self.mean = mean
        self.std = std
        self.pixelDiv = pixelDiv
        self.input_size = input_size 
        self.paths = []
        image_paths = os.path.join(rootDir, 'left_frames')
        image_left_paths = glob.glob(image_paths + "/*.png")
        groundtruth_paths_left_foregrasp = os.path.join(rootDir, 'ground_truth', 'Left_Prograsp_Forceps_labels')
        ground_truth_maryland = os.path.join(rootDir, 'ground_truth', 'Maryland_Bipolar_Forceps_labels')
        groundtruth_other = os.path.join(rootDir, 'ground_truth', 'Other_labels')
        groundtruth_right_foregrasp =  os.path.join(rootDir, 'ground_truth', 'Right_Prograsp_Forceps_labels') 

        for i in image_left_paths:
            head, tail = os.path.split(i)
            im_l_lfrg = groundtruth_paths_left_foregrasp + '/' + tail
            im_mary = ground_truth_maryland + '/' + tail
            im_oth = groundtruth_other + '/' + tail
            im_r_lfrg = groundtruth_right_foregrasp + '/' + tail
            
            if os.path.isfile(im_l_lfrg) and os.path.isfile(im_mary) and os.path.isfile(im_oth) and os.path.isfile(im_r_lfrg):
                self.paths.append((i, im_l_lfrg, im_mary, im_oth, im_r_lfrg))
      
    def normalize_image(self, img):
        return (img.astype(np.float32) / self.pixelDiv)
        
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image_src = cv2.resize(cv2.cvtColor(cv2.imread(self.paths[idx][0]), cv2.COLOR_BGR2RGB), self.input_size)
        image_l_prog = cv2.resize(cv2.cvtColor(cv2.imread(self.paths[idx][1]), cv2.COLOR_BGR2GRAY), self.input_size)
        image_mary = cv2.resize(cv2.cvtColor(cv2.imread(self.paths[idx][2]), cv2.COLOR_BGR2GRAY), self.input_size)
        image_oth = cv2.resize(cv2.cvtColor(cv2.imread(self.paths[idx][3]), cv2.COLOR_BGR2GRAY), self.input_size)
        image_r_prog = cv2.resize(cv2.cvtColor(cv2.imread(self.paths[idx][4]), cv2.COLOR_BGR2GRAY), self.input_size)
        
        image_mask = torch.from_numpy(self.normalize_image(np.array([image_l_prog, image_mary, image_oth, image_r_prog])))
        image_src = torch.from_numpy(self.normalize_image(np.transpose(image_src, (2, 0, 1))))
        return image_src, image_mask
        


        
        