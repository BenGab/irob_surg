# -*- coding: utf-8 -*-
import numpy as np

def recon_from_disp(disparity, img_opp):
    h, w = disparity.shape
    im_rec = np.zeros_like(disparity)
    for y in range(h):    
        for x in range(w):
           intense = disparity[y,x]
           x_range = x + intense
           if x_range >= 0 and x_range < w:
               im_rec[y, x] = img_opp[y, x_range]
    return im_rec

def calc_loss_img_diff(img_recon, img_expected):
    h, w = img_expected.shape
    avg_val = 1/(h * w)
    diff = 0
    for y in range(h):    
        for x in range(w):
            diff = diff + (np.abs((img_expected[y, x] - img_recon[y, x])))
            
    return (avg_val * diff)
