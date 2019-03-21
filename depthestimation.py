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
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64, 1e-3),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64, 1e-3))
        
        self.cnn2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128, 1e-3),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128, 1e-3))
        
        self.cnn3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3))
        
        self.cnn4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3))
        
        self.cnn5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3))
        
        self.cnn6 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512))
        
        self.decnn5 = nn.Sequential(
                    nn.ConvTranspose2d(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.ConvTranspose2d(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3),
                    nn.ConvTranspose2d(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512, 1e-3))
                
        self.decnn4 = nn.Sequential(
                nn.ConvTranspose2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512, 1e-3),
                nn.ConvTranspose2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512, 1e-3),
                nn.ConvTranspose2d(512, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3))
        
        self.decnn3 = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3),
                nn.ConvTranspose2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256, 1e-3),
                nn.ConvTranspose2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128, 1e-3))
        
        self.decnn2 = nn.Sequential(
                nn.ConvTranspose2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128, 1e-3),
                nn.ConvTranspose2d(128, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64, 1e-3))
        
        self.decnn1 = nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(64, 1e-3),
                    nn.ConvTranspose2d(64, 3, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(3, 1e-3),
                    nn.Conv2d(3, 1, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(1, 1e-3))

    def calc_recon_from_disp(self, depth_est, img_from_rec):
        dp_im = depth_est.permute(0, 3, 2, 1).abs() * 255
        img_from_rec = img_from_rec.permute(0, 3, 2, 1)
        im_rec = torch.zeros_like(img_from_rec, requires_grad=True)
        b, h, w, d = depth_est.size()
        for batch in range(b):
            for y in range(h):
                for x in range(w):
                    disp_im_idx = dp_im[batch][y][x].detach().numpy() + x
                    im_rec[batch][y][x][0] = 0
                    im_rec[batch][y][x][1] = 0
                    im_rec[batch][y][x][2] = 0
                    if (int(disp_im_idx) < w):
                        im_rec[batch][y][x][0] = img_from_rec[batch][y][int(disp_im_idx)][0]
                        im_rec[batch][y][x][1] = img_from_rec[batch][y][int(disp_im_idx)][1]
                        im_rec[batch][y][x][2] = img_from_rec[batch][y][int(disp_im_idx)][2]
        return dp_im


    def forward_part(self, img):
        MP = nn.MaxPool2d(2, stride=2, return_indices=True)
        UP = nn.MaxUnpool2d(2, stride=2) 
        #encoder
        disp_est = self.cnn1(img)
        disp_est, unpool_1 = MP(disp_est)

        disp_est = self.cnn2(disp_est)
        disp_est, unpool_2 = MP(disp_est)
        # print(disp_est.size())

        disp_est = self.cnn3(disp_est)
        disp_est, unpool_3 = MP(disp_est)
        # print(disp_est.size())
        
        disp_est = self.cnn4(disp_est)
        disp_est, unpool_4 = MP(disp_est)
        # print(disp_est.size())
        
        disp_est = self.cnn5(disp_est)
        disp_est, unpool_5 = MP(disp_est)
        # print(disp_est.size())
        
        disp_est = self.cnn6(disp_est)
        disp_est = UP(disp_est, unpool_5)
        # print(disp_est.size())
        
        #decoder
        disp_est = self.decnn5(disp_est)
        disp_est = UP(disp_est, unpool_4)
        # print(disp_est.size())
        
        disp_est = self.decnn4(disp_est)
        disp_est = UP(disp_est, unpool_3)
        # print(disp_est.size())
        
        disp_est = self.decnn3(disp_est)
        disp_est = UP(disp_est, unpool_2)
        # print(disp_est.size())

        disp_est = self.decnn2(disp_est)
        disp_est = UP(disp_est, unpool_1)
        # print(disp_est.size())
        
        disp_est = self.decnn1(disp_est)
        # print(disp_est.size())
        return disp_est
    
    def forward(self, img_l, img_r):
        disp_est_l = self.forward_part(img_l)
        # disp_est_r = self.forward_part(img_r)
        im_rec = self.calc_recon_from_disp(disp_est_l, img_r)
        return disp_est_l, im_rec