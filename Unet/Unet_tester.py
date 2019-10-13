import torch
import torch.nn as nn
from Unet import UNet
from UnetDataset import ToolDataset
from NetUtils import save_model

input_size = (256, 256)
mean = 0.485
std = 0.229
pixdeviator = 255
dataset = ToolDataset(['/home/bennyg/Development/datasets/miccai_challenge_2018_release_1/seq_1',
                       '/home/bennyg/Development/datasets/miccai_challenge_2018_release_1/seq_2',
                       '/home/bennyg/Development/datasets/miccai_challenge_2018_release_1/seq_3',
                       '/home/bennyg/Development/datasets/miccai_challenge_2018_release_1/seq_4',
                       '/home/bennyg/Development/datasets/miccai_challenge_release_2/seq_5',
                       '/home/bennyg/Development/datasets/miccai_challenge_release_2/seq_6',
                       '/home/bennyg/Development/datasets/miccai_challenge_release_2/seq_7',
                       '/home/bennyg/Development/datasets/miccai_challenge_release_3/seq_9',
                       '/home/bennyg/Development/datasets/miccai_challenge_release_3/seq_10',
                       '/home/bennyg/Development/datasets/miccai_challenge_release_3/seq_11',
                       '/home/bennyg/Development/datasets/miccai_challenge_release_3/seq_12'
                       '/home/bennyg/Development/datasets/miccai_challenge_release_4/seq_13', 
                       '/home/bennyg/Development/datasets/miccai_challenge_release_4/seq_14',
                       '/home/bennyg/Development/datasets/miccai_challenge_release_4/seq_15',
                       '/home/bennyg/Development/datasets/miccai_challenge_release_4/seq_16'], mean, 
                      std, pixdeviator, input_size)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

number_epoch = 5
learning_rate = 1e-3
net = UNet(1)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
number_train = len(dataloader)
for i in range(number_epoch):
    epoch_loss = 0.0
    loss_index = 0
    net.train()
    for image_src, image_mask in dataloader:
        loss_index += 1
        pred_mask = net(image_src)
        pred_flat = torch.flatten(pred_mask, start_dim=1)
        mask_flat = torch.flatten(image_mask, start_dim=1)
        loss = criterion(pred_mask, image_mask)
        epoch_loss += loss.item()
        print('{0}/{1} --- loss: {2:.6f}'.format(loss_index, number_train, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
          
    print('Epoch: {} / {} Loss: {}'.format(i + 1, number_epoch, epoch_loss / loss_index))
    
protoPath = '/home/bennyg/Development/pretrained_models/Unet_C1.pt'
save_model(net, protoPath)
