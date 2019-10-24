# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import glob
    
class MiccaiDataset:
    def __init__(self, rootDirs, pixelDiv, input_size, batch_size=1):
        self.pixelDiv = pixelDiv
        self.input_size = input_size 
        self.batch_size = batch_size
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
            image_src = cv2.resize(cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2GRAY), self.input_size)
            label_src = cv2.resize(cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2GRAY), self.input_size).reshape(256, 256, 1)
            
            self.images.append(self.normalize_image(image_src))
            self.labels.append(self.normalize_image(label_src))
        
        return np.array(self.images), np.array(self.labels)

    def data_len(self):
        return len(self.paths) / self.batch_size
    
    def image_generator(self):
        while True:
            start=0
            end=self.batch_size 
            images, labels = [], []
            for i in range(start, end):
                image, mask = self.paths[i]
                image_src = cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY), self.input_size)
                label_src = cv2.resize(cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY), self.input_size)
                images.append(self.normalize_image(image_src).reshape(256, 256, 1))
                labels.append(self.normalize_image(label_src).reshape(256, 256, 1))
            start+=self.batch_size
            end+=self.batch_size              
            yield np.array(images), np.array(labels)


            
        
        
    
