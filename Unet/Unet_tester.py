import torch
import torch.nn as nn
from Unet import UNet
from UnetDataset import ToolDataset
from NetUtils import save_model

input_size = (256, 256)
mean = 0.485
std = 0.229
pixdeviator = 255
dataset = ToolDataset('/home/bennyg/Development/datasets/instrument_dataset_1', mean, 
                      std, pixdeviator, input_size)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
number_epoch = 5
learning_rate = 1e-1
net = UNet(4)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
number_train = len(dataloader)
for i in range(number_epoch):
    epoch_loss = 0.0
    loss_index = 0
    net.train()
    for image_src, image_mask in dataloader:
        loss_index += 1
        pred_mask = net(image_src)
        prob_mask_flat = pred_mask.view(-1)
        true_mask_flat = image_mask.view(-1)
        loss = criterion(prob_mask_flat, true_mask_flat)
        epoch_loss += loss.item()
        print('{0}/{1} --- loss: {2:.6f}'.format(loss_index, number_train, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
          
    print('Epoch: {} / {} Loss: {0:.4f}'.format(i + 1, number_epoch, epoch_loss / loss_index))
    
protoPath = '/home/bennyg/Development/pretrained_models/Unet_C4.pt'
save_model(net, protoPath)
