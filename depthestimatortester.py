# -*- coding: utf-8 -*-

import torch
import depthestimation as DE
import davinci_dataset as dvs
from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size=16
train_data = dvs.DavinciDataset("/home/bennyg/Development/DataSet/daVinci", True, 'image_0', 'image_1') 
testdataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
net = DE.DepthEstimatorNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

for image_l, image_r in testdataloader:
    print(image_l.shape)
    dept_l, dept_r = net(image_l.to(device), image_r.to(device))
    print(dept_l.shape)
    print(dept_r.shape)
    break
