# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import glob
import shutil
import random
    
class MiccaiDataset:
    def __init__(self, rootDir, pixelDiv, input_size, batch_size=1):
        self.pixelDiv = pixelDiv
        self.input_size = input_size 
        self.batch_size = batch_size
        self.paths = []
        self.images = []
        self.labels = []

        image_paths = os.path.join(rootDir, 'images')
        image_paths = glob.glob(image_paths + "/*.jpg")
        labels_path = os.path.join(rootDir, 'masks')  
        for i in image_paths:
            head, tail = os.path.split(i)
            im_label = labels_path + '/' + tail              
            if os.path.isfile(im_label):
                self.paths.append((i, im_label))
        self.val_paths = self.paths[0:115]
        self.train_paths = self.paths[116::]
                    
                    
    def normalize_image(self, img):
        return (img.astype(np.float32) / self.pixelDiv)
    
    def load_images(self):
        for src_path, label_path in self.paths:
            image_src = cv2.resize(cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB), self.input_size)
            label_src = cv2.resize(cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB), self.input_size)
            
            self.images.append(self.normalize_image(image_src))
            self.labels.append(self.normalize_image(label_src))
        
        return np.array(self.images), np.array(self.labels)

    def data_train_len(self):
        return len(self.train_paths) // self.batch_size
    
    def data_val_len(self):
        return len(self.val_paths) // self.batch_size

    def copy_images(self, folder_img, folder_labels):
        for i, (image, mask) in enumerate(self.paths, start=0):
            name = "000{}.png"
            if i >= 10:
                name="00{}.png"
            if i >= 100:
                name="0{}.png" 
            if i >= 1000:
                name="{}.png"
            filename = name.format(i)
            shutil.copy(image, '{}/{}'.format(folder_img, filename))
            shutil.copy(mask, '{}/{}'.format(folder_labels, filename))

    
    def image_train_generator(self):
        while True:
            start=0
            end=self.batch_size 
            images, labels = [], []
            path = self.train_paths
            for i in range(start, end):
                image, mask = path[i]
                image_src = cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), self.input_size)
                label_src = cv2.resize(cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB), self.input_size)
                images.append(self.normalize_image(image_src))
                labels.append(self.normalize_image(label_src))
            start+=self.batch_size
            end+=self.batch_size              
            yield np.array(images), np.array(labels)
            
    def image_val_generator(self):
        while True:
            start=0
            end=self.batch_size 
            images, labels = [], []
            for i in range(start, end):
                image, mask = self.val_paths[i]
                image_src = cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), self.input_size)
                label_src = cv2.resize(cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB), self.input_size)
                images.append(self.normalize_image(image_src))
                labels.append(self.normalize_image(label_src))
            start+=self.batch_size
            end+=self.batch_size              
            yield np.array(images), np.array(labels)


            
        
        
    
