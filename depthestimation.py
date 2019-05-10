# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:37:46 2018

@author: Benkő Gábor
"""

import torch
import torch.nn as nn
from torch.nn.functional import pad

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

    def image_warp(img, depth, padding_mode='zeros'):

        # img: the source image (where to sample pixels) -- [B, 3, H, W]
        # depth: depth map of the target image -- [B, 1, H, W]
        # Returns: Source image warped to the target image

        b, _, h, w = depth.size()

        # i_range = torch.linspace(0, w - 1, w).repeat(h, 1)
        i_range = torch.autograd.Variable(torch.linspace(0, w-1, w).view(1, h, 1).expand(1, h, w),
                                          requires_grad=False)  # [1, H, W]  copy 0-height for w times : y coord
        j_range = torch.autograd.Variable(torch.linspace(-1.0, 1.0).view(1, 1, w).expand(1, h, w),
                                          requires_grad=False)  # [1, H, W]  copy 0-width for h times  : x coord

        pixel_coords = torch.stack((j_range, i_range), dim=1).float().cuda()  # [1, 2, H, W]
        batch_pixel_coords = pixel_coords[:, :, :, :].expand(b, 2, h, w).contiguous().view(b, 2, -1)  # [B, 2, H*W]

        X = batch_pixel_coords[:, 0, :] + depth.contiguous().view(b, -1)  # [B, H*W]
        Y = batch_pixel_coords[:, 1, :]

        X_norm = X
        Y_norm = Y

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        pixel_coords = pixel_coords.view(b, h, w, 2)  # [B, H, W, 2]

        projected_img = torch.nn.functional.grid_sample(img, pixel_coords, padding_mode=padding_mode)

        return projected_img

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
        return disp_est_l