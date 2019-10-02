#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:34:49 2018

@author: bennyg
"""

import cv2
from matplotlib import pyplot as plt
import torch
from functions import rebuild_from_disparity
import numpy as np

left_image_path = 'dataset/instrument_dataset_1/left_frames/frame000.png'
right_mage_path = 'dataset/instrument_dataset_1/right_frames/frame000.png'

img_l = cv2.imread(left_image_path)
img_r = cv2.imread(right_mage_path)
img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create()

disparity = stereo.compute(img_l, img_r)
plt.imshow(disparity, 'gray')
plt.show()

plt.imshow(img_l, 'gray')
plt.show()

plt.imshow(img_r, 'gray')
plt.show()


left_tensor = torch.from_numpy(img_l)
right_tensor = torch.from_numpy(img_r)
disp_tensor = torch.from_numpy(disparity)

recon = rebuild_from_disparity(right_tensor, disp_tensor)

print(disp_tensor)


          