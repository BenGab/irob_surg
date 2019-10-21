# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import glob
    
class MiccaiDataset:
    def __init__(self, rootDirs, pixelDiv, input_size):
        self.pixelDiv = pixelDiv
        self.input_size = input_size 
        self.paths = []
        self.images = []
        self.labels = []
        for rootDir in rootDirs:
            image_paths = os.path.join(rootDir, 'left_frames')
            image_left_paths = glob.glob(image_paths + "/*.png")
            labels_path = os.path.join(rootDir, 'labels')  
            for i in image_left_paths:
                head, tail = os.path.split(i)
                im_label = labels_path + '/' + tail              
                if os.path.isfile(im_label):
                    self.paths.append((i, im_label))
                    
                    
    def normalize_image(self, img):
        return (img.astype(np.float32) / self.pixelDiv)
    
    def load_images(self):
        for src_path, label_path in self.paths:
            image_src = cv2.resize(cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB), self.input_size)
            label_src = cv2.resize(cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB), self.input_size)
            
            self.images.append(self.normalize_image(image_src))
            self.labels.append(self.normalize_image(label_src))
        
        return np.array(self.images), np.array(self.labels)

    def data_len(self):
        return len(self.paths)
    
    def image_generator(self):
        for image, mask in self.paths:           
            image_src = cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), self.input_size)
            label_src = cv2.resize(cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB), self.input_size)
            yield (np.array([self.normalize_image(image_src)]), np.array([self.normalize_image(label_src)]))



            
        
        
    
