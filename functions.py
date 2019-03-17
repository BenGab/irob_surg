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

def calc_loss_img_diff(disp_im, img_from_rec, img_expected, weight = 0.5):
    b, h, w, d = disp_im.size()
    im_rec = torch.zeros_like(img_expected, requires_grad=True)
    loss_batch = 0
    for batch in range(b):
        for y in range(h):
            for x in range(w):
                disp_im_idx = disp_im[batch][y][x].detach().numpy() + x
                im_rec[batch][y][x][0] = 0
                im_rec[batch][y][x][1] = 0
                im_rec[batch][y][x][2] = 0
                if(int(disp_im_idx) < w):
                    im_rec[batch][y][x][0] = img_from_rec[batch][y][int(disp_im_idx)][0]
                    im_rec[batch][y][x][1] = img_from_rec[batch][y][int(disp_im_idx)][1]
                    im_rec[batch][y][x][2] = img_from_rec[batch][y][int(disp_im_idx)][2]
        loss = 1/(h*w) * (torch.sum(img_expected - im_rec).abs())
        loss_batch = loss_batch + loss
    return loss_batch * weight



def calc_loss_img_disp(disp_est_l, disp_est_r, weight = 1.0):
    b, h, w, d = disp_est_l.size()
    dep_r = torch.zeros((b, h, w, d))
    loss_batch = 0
    for batch in range(b):
        for y in range(h):
            for x in range(w):
                disp_r_idx = dep_r[batch][y][x].detach().numpy() + x
                dep_r[batch][y][x] = 0
                if(int(disp_r_idx) < w):
                    dep_r[batch][y][x] = disp_est_r[batch][y][int(disp_r_idx)]
        loss = 1/(h*w) * (torch.sum(disp_est_l - dep_r).abs())
        loss_batch = loss_batch + loss
    return loss_batch * weight

def loss_fn(dep_l, dep_r, im_l, im_r, device):
    dep_l = dep_l.permute(0, 3, 2, 1).abs() * 255
    dep_r = dep_r.permute(0, 3, 2, 1).abs() * 255
    im_l = im_l.permute(0, 3, 2, 1)
    im_r = im_r.permute(0, 3, 2, 1)
    dep_diff = calc_loss_img_disp(dep_l, dep_r)
    rec_diff_l = calc_loss_img_diff(dep_l, im_r, im_l)
    rec_diff_r = calc_loss_img_diff(dep_r, im_l, im_r)
    return rec_diff_l + rec_diff_r + dep_diff
    
    
    