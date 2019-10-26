# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

def mask_overlay(image, mask, color=(0, 255, 0)):
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

image_p = '/home/bennyg/Development/datasets/miccai/dataset/images/frame_0_1.jpg'
mask_p = '/home/bennyg/Development/datasets/UNET12_BCE_G_C2.jpg'

img=cv2.resize(cv2.cvtColor(cv2.imread(image_p), cv2.COLOR_BGR2RGB), (256, 256))
mask=cv2.cvtColor(cv2.imread(mask_p), cv2.COLOR_BGR2GRAY)

overlay = mask_overlay(img, mask)

plt.imshow(overlay)