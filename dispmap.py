#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:34:49 2018

@author: bennyg
"""

import cv2
from matplotlib import pyplot as plt
import functions as fn
    

img_l = cv2.imread('im_l.png')
img_r = cv2.imread('im_r.png')
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

#reconstruct
img_rc = fn.recon_from_disp(disparity, img_r) 
loss = fn.calc_loss_img_diff(img_rc, img_l)

print(loss)

plt.imshow(img_rc, 'gray')
plt.show()          