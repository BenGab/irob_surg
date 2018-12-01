# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torchvision.transforms.functional as tfn
import numpy as np
import depthestimation as DE
import davinci_dataset as dvs
import functions as lrf
from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size=16
train_data = dvs.DavinciDataset("/home/bennyg/Development/DataSet/daVinci", True, 'image_0', 'image_1') 
testdataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
net = DE.DepthEstimatorNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

for image_l, image_r in testdataloader:
    image_l = image_l.to(device)
    image_r = image_r.to(device)
    optimizer.zero_grad()
    dept_l, dept_r = net(image_l, image_r)
    loss = lrf.loss_fn(dept_l, dept_r, image_l, image_r, device)
    loss.backward()
    print(loss)
    optimizer.step()
    break
