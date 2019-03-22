# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import depthestimation as DE
import davinci_dataset as dvs

device = "cpu"
batch_size=16
train_data = dvs.DavinciDataset("d:\/Development/daVinci/daVinci/", True, 'image_0', 'image_1')
testdataloader = torch.utils.data.DataLoader(train_data, batch_size=25, shuffle=True)
net = DE.DepthEstimatorNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
criterion = nn.MSELoss()
for epoch in range(40):
    i = 0
    running_loss = 0.0
    for image_l, image_r, image_rg in testdataloader:
        image_l = image_l.to(device)
        image_r = image_r.to(device)
        dept_l, im_rec = net(image_l, image_r, image_rg)
        optimizer.zero_grad()
        image_l = image_l.permute(0, 1, 3, 2)
        loss = criterion(im_rec, image_l)
        print(loss)
        loss.backward()
        optimizer.step()
        i += 1
        running_loss += loss.item()
        torch.cuda.empty_cache()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

