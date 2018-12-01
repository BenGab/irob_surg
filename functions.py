# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F

def recon_from_disp_cv(disparity, img_opp):
    h, w = disparity.shape
    im_rec = np.zeros_like(disparity)
    for y in range(h):    
        for x in range(w):
           intense = disparity[y,x]
           x_range = x + intense
           if x_range >= 0 and x_range < w:
               im_rec[y, x] = img_opp[y, x_range]
    return im_rec

def calc_loss_img_diff_cv(img_recon, img_expected, weight = 0.5):
    h, w = img_expected.shape
    avg_val = 1/(h * w)
    diff = 0
    for y in range(h):    
        for x in range(w):
            diff = diff + (np.abs((img_expected[y, x] - img_recon[y, x])))            
    return weight * (avg_val * diff)

def calc_loss_img_disp_cv(disp_est_l, disp_est_r, weight = 1.0):
    h, w = disp_est_l.shape
    avg_val = 1/(h * w)
    diff = 0
    for i in range(h):
        for j in range(w):
            diff = diff + (np.abs(disp_est_l[i,j] - disp_est_r[(i + disp_est_r[i,j]), j]))
    return  weight * (avg_val * diff)

def recon_from_disp_pil(disparity, img_opp):
    _, h, w, _ = disparity.shape
    recon = torch.zeros_like(img_opp)
    for y in range(h):    
        for x in range(w):
           intense = disparity[0, y, x]
           intense = (intense.detach().cpu().numpy()[0])
           x_range = x + np.int32(intense)
           if x_range >= 0 and x_range < w:
               #print(img_opp[0, y, x_range])
               recon[0, y, x] = img_opp[0, y, x_range]
    return recon

def calc_loss_img_diff_pil(img_recon, img_expected, weight = 0.5):
    _, h, w, _ = img_recon.shape
    recon_sum = img_recon[0].sum().abs()
    expected_sum = img_expected[0].sum().abs()   
    rec_diff = (recon_sum - expected_sum) * (1/(h*w)) * weight        
    return rec_diff.abs()

def calc_loss_img_disp_pil(disp_est_l, disp_est_r, weight = 1.0):
    _, h, w, _ = disp_est_l.shape
    dep_sum_l = disp_est_l[0].sum().abs()
    dep_sum_r = disp_est_r[0].sum().abs()
    dep_diff = (dep_sum_l - dep_sum_r) * (1/(h*w)) * weight
    return dep_diff.abs()

def loss_fn(dep_l, dep_r, im_l, im_r, device):
    dep_l = dep_l.permute(0, 2, 3, 1)
    dep_r = dep_r.permute(0, 2, 3, 1)
    im_l = im_l.permute(0, 2, 3, 1)
    im_r =im_r.permute(0, 2, 3, 1)
    #print(dep_l.shape)
    #print(dep_r.shape)
    #print(im_l.shape)
    #print(im_r.shape)
    dep_diff = calc_loss_img_disp_pil(dep_l, dep_r)
    recon_tensor_l = recon_from_disp_pil(dep_l, im_r)
    recon_tensor_r = recon_from_disp_pil(dep_r, im_l)
    rec_diff_l = calc_loss_img_diff_pil(recon_tensor_l, im_l)
    rec_diff_r = calc_loss_img_diff_pil(recon_tensor_r, im_r)
    return dep_diff + rec_diff_l + rec_diff_r
    
    
    