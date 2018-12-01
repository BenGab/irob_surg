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
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, stride=2))
        
        self.cnn2 = nn.Sequential(
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2))
        
        self.cnn3 = nn.Sequential(
                nn.Conv2d(128, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(2, stride=2))
        
        self.cnn4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(256),
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.MaxPool2d(2, stride=2))
        
        self.cnn5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.Conv2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.MaxPool2d(2, stride=2))
        
        self.cnn6 = nn.Sequential(
                nn.Conv2d(512, 512, 3),
                nn.ReLU(),
                nn.BatchNorm2d(512))
        
        self.decnn5 = nn.Sequential(
                    nn.MaxUnpool2d(2, stride=2),
                    nn.ConvTranspose2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.ConvTranspose2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.ConvTranspose2d(512, 512, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.MaxUnpool2d(2, stride=2))
                
        self.decnn4 = nn.Sequential(
                nn.ConvTranspose2d(512, 512, 3),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.ConvTranspose2d(512, 512, 3),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.ConvTranspose2d(512, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxUnpool2d(2, stride=2))
        
        self.decnn3 = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.ConvTranspose2d(256, 256, 3),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.ConvTranspose2d(256, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxUnpool2d(2, stride=2))
        
        self.decnn2 = nn.Sequential(
                nn.ConvTranspose2d(128, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 64, 3),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxUnpool2d(2, stride=2))
        
        self.decnn1 = nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.ConvTranspose2d(64, 3, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(3),
                    nn.Conv2d(3, 1, 3))
        
    def forward_part(self, img):
        #encoder
        disp_est = self.cnn1(img)
        disp_est = self.cnn2(disp_est)
        disp_est = self.cnn3(disp_est)
        disp_est = self.cnn4(disp_est)
        disp_est = self.cnn5(disp_est)
        disp_est = self.cnn6(disp_est)
        #decoder
        disp_est = self.decnn5(disp_est)
        disp_est = self.decnn4(disp_est)
        disp_est = self.decnn3(disp_est)
        disp_est = self.decnn2(disp_est)
        disp_est = self.decnn1(disp_est)
        return disp_est
    
    def forward(self, img_l, img_r):
        disp_est_l = self.forward_part(img_l)
        disp_est_r = self.forward_part(img_r)
        return disp_est_l, disp_est_r 