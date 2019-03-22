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

    def apply_disparity(self, input_images, x_offset, wrap_mode='border', tensor_type='torch.FloatTensor'):
        num_batch, num_channels, width, height = input_images.size()

        # Handle both texture border types
        edge_size = 0
        if wrap_mode == 'border':
            edge_size = 1
            # Pad last and second-to-last dimensions by 1 from both sides
            input_images = pad(input_images, (1, 1, 1, 1))
        elif wrap_mode == 'edge':
            edge_size = 0
        else:
            return None

        # Put channels to slowest dimension and flatten batch with respect to others
        input_images = input_images.permute(1, 0, 2, 3).contiguous()
        im_flat = input_images.view(num_channels, -1)

        # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
        # meshgrid function)
        x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type)
        y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type)
        # Take padding into account
        x = x + edge_size
        # y = y + edge_size
        # Flatten and repeat for each image in the batch
        x = x.view(-1).repeat(1, num_batch)
        y = y.contiguous().view(-1).repeat(1, num_batch)

        # Now we want to sample pixels with indicies shifted by disparity in X direction
        # For that we convert disparity from % to pixels and add to X indicies
        x = x + x_offset.contiguous().view(-1) * width
        # Make sure we don't go outside of image
        x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
        # Round disparity to sample from integer-valued pixel grid
        y0 = torch.floor(y)
        # In X direction round both down and up to apply linear interpolation
        # between them later
        x0 = torch.floor(x)
        x1 = x0 + 1
        # After rounding up we might go outside the image boundaries again
        x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

        # Calculate indices to draw from flattened version of image batch
        dim2 = (width + 2 * edge_size)
        dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
        # Set offsets for each image in the batch
        base = dim1 * torch.arange(num_batch).type(tensor_type)
        base = base.view(-1, 1).repeat(1, height * width).view(-1)
        # One pixel shift in Y  direction equals dim2 shift in flattened array
        base_y0 = base + y0 * dim2
        # Add two versions of shifts in X direction separately
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        # Sample pixels from images
        pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
        pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

        # Apply linear interpolation to account for fractional offsets
        weight_l = x1 - x
        weight_r = x - x0
        output = weight_l * pix_l + weight_r * pix_r

        # Reshape back into image batch and permute back to (N,C,H,W) shape
        output = output.view(num_channels, num_batch, height, width).permute(1, 0, 2, 3)

        return output

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
    
    def forward(self, img_l, img_r, image_rg):
        disp_est_l = self.forward_part(img_l)
        # disp_est_r = self.forward_part(img_r)
        im_rec = self.apply_disparity(img_r, disp_est_l)
        return disp_est_l, im_rec