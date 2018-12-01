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
    h, w = disparity.shape
    im_rec = np.zeros_like(img_opp)
    for y in range(h):    
        for x in range(w):
           intense = disparity[y,x]
           x_range = x + intense
           if x_range >= 0 and x_range < w:
               im_rec[y, x] = img_opp[y, x_range]
    return im_rec

def calc_loss_img_diff_pil(img_recon, img_expected, weight = 0.5):
    h, w, _ = img_recon.shape
    avg_val = 1/(h * w)
    diff = 0
    for y in range(h):    
        for x in range(w):
            exp_sum = np.sum(img_expected[y, x])
            recon_sum = np.sum(img_recon[y, x])
            diff = diff + (np.abs(exp_sum - recon_sum))            
    return weight * (avg_val * diff)

def calc_loss_img_disp_pil(disp_est_l, disp_est_r, weight = 1.0):
    h, w  = disp_est_l.shape
    avg_val = 1/(h * w)
    diff = 0
    for i in range(h):
        for j in range(w):
            diff = diff + (np.abs(disp_est_l[i,j] - disp_est_r[i, j]))
    return weight * (avg_val * diff)

def loss_fn_pil(dep_l, dep_r, im_l, im_r):
    l_recon = recon_from_disp_pil(np.array(dep_l), np.array(im_r))
    r_recon = recon_from_disp_pil(np.array(dep_r), np.array(im_l))
    lc_dep = calc_loss_img_disp_pil(np.array(dep_l), np.array(dep_r))
    lf_l = calc_loss_img_diff_pil(l_recon, np.array(im_l))
    lf_r = calc_loss_img_diff_pil(r_recon, np.array(im_r))
    return lc_dep + lf_l + lf_r