# -*- coding: utf-8 -*-

import torch
import depthestimation as DE
import davinci_dataset as dvs
import functions as lrf

device = "cpu"
batch_size=16
train_data = dvs.DavinciDataset("d:\/Development/daVinci/daVinci/", True, 'image_0', 'image_1')
testdataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
net = DE.DepthEstimatorNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

for epoch in range(1):
    i = 0
    running_loss = 0.0
    for image_l, image_r in testdataloader:
        image_l = image_l.to(device)
        image_r = image_r.to(device)
        print(image_l.shape)
        optimizer.zero_grad()
        dept_l, dept_r = net(image_l, image_r)
        loss = lrf.loss_fn(dept_l, dept_r, image_l, image_r, device)
        print(loss)
        loss.backward()
        optimizer.step()
        i += 1
        running_loss += loss.item()
        #torch.cuda.empty_cache()
        # print(i)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


