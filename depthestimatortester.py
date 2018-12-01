# -*- coding: utf-8 -*-

import torch
import torchvision.transforms.functional as tfn
import numpy as np
import depthestimation as DE
import davinci_dataset as dvs
import functions as lrf
from matplotlib import pyplot as plt
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size=16
train_data = dvs.DavinciDataset("/home/bennyg/Development/DataSet/daVinci", True, 'image_0', 'image_1') 
testdataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
net = DE.DepthEstimatorNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

for image_l, image_r in testdataloader:
    dept_l, dept_r = net(image_l.to(device), image_r.to(device))
    im_l = tfn.to_pil_image(image_l[0].cpu().data)
    im_r = tfn.to_pil_image(image_r[0].cpu().data)
    dep_l = tfn.to_pil_image(dept_l[0].cpu().data)
    dep_r = tfn.to_pil_image(dept_r[0].cpu().data)
    
    loss = lrf.loss_fn_pil(dep_l, dep_r, im_l, im_r)
    print(loss)
    break
