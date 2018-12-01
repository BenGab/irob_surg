# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:37:46 2018

@author: Benkő Gábor
"""

import torch
import torch.nn as nn

class DepthEstimatorNet(nn.Module):
    def __init__(self):
        super(DepthEstimatorNet, self).__init__()
        
        self.cnn1 = nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.ReLU(),
                nn.BatchNorm2d(64, 1e-3),
                nn.Conv2d(64, 64, 3),
                nn.ReLU(),
                nn.BatchNorm2d(64, 1e-3))
        
        self.cnn2 = nn.Sequential(
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128, 1e-3),
                nn.Conv2d(128, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128, 1e-3))
        
        self.cnn3 = nn.Sequential(
                nn.Conv2d(128, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3),
                nn.Conv2d(256, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3),
                nn.Conv2d(256, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3))
        
        self.cnn4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3))
        
        self.cnn5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3))
        
        self.cnn6 = nn.Sequential(
                nn.Conv2d(512, 512, 3),
                nn.ReLU(),
                nn.BatchNorm2d(512))
        
        self.decnn5 = nn.Sequential(
                    nn.ConvTranspose2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.ConvTranspose2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.ConvTranspose2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3))
                
        self.decnn4 = nn.Sequential(
                nn.ConvTranspose2d(512, 512, 3),
                nn.ReLU(),
                nn.BatchNorm2d(512, 1e-3),
                nn.ConvTranspose2d(512, 512, 3),
                nn.ReLU(),
                nn.BatchNorm2d(512, 1e-3),
                nn.ConvTranspose2d(512, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3))
        
        self.decnn3 = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3),
                nn.ConvTranspose2d(256, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3),
                nn.ConvTranspose2d(256, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128, 1e-3))
        
        self.decnn2 = nn.Sequential(
                nn.ConvTranspose2d(128, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128, 1e-3),
                nn.ConvTranspose2d(128, 64, 3),
                nn.ReLU(),
                nn.BatchNorm2d(64, 1e-3))
        
        self.decnn1 = nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(64, 1e-3),
                    nn.ConvTranspose2d(64, 3, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(3, 1e-3),
                    nn.Conv2d(3, 1, 3))
        
    def forward_part(self, img):
        MP = nn.MaxPool2d(2, stride=2, return_indices=True)
        UP = nn.MaxUnpool2d(2, stride=2) 
        #encoder
        disp_est = self.cnn1(img)
        #disp_est, indices_1 = MP(disp_est)
        
        disp_est = self.cnn2(disp_est)
        #disp_est, indices_2 = MP(disp_est)
        
        disp_est = self.cnn3(disp_est)
        #disp_est, indices_3 = MP(disp_est)
        
        disp_est = self.cnn4(disp_est)
        #disp_est, indices_4 = MP(disp_est)
        
        disp_est = self.cnn5(disp_est)
        #disp_est, indices_5 = MP(disp_est)
        
        disp_est = self.cnn6(disp_est)
        #disp_est = UP(disp_est, indices_5)
        
        #decoder
        disp_est = self.decnn5(disp_est)
        #disp_est = UP(disp_est, indices_4)
        
        disp_est = self.decnn4(disp_est)
        #disp_est = UP(disp_est, indices_3)
        
        disp_est = self.decnn3(disp_est)
        #disp_est = UP(disp_est, indices_2)
        
        disp_est = self.decnn2(disp_est)
        #disp_est = UP(disp_est, indices_1)
        
        disp_est = self.decnn1(disp_est)
        return disp_est
    
    def forward(self, img_l, img_r):
        disp_est_l = self.forward_part(img_l)
        disp_est_r = self.forward_part(img_r)
        return disp_est_l.type(torch.IntTensor), disp_est_r.type(torch.IntTensor)
    