import torch
from Unet import UNet
from UnetDataset import ToolDataset

input_size = (256, 256)
mean = 0.485
std = 0.229
pixdeviator = 255
dataset = ToolDataset('/home/bennyg/Development/datasets/instrument_dataset_1', mean, 
                      std, pixdeviator, input_size)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

net = UNet(4)

for image_src, image_mask in dataloader:
    pred_mask = net(image_src)
    print(pred_mask.shape)
    break


