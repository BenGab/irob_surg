import UnetDataset as uds

input_size = (256, 256)
mean = 0.485
std = 0.229
pixdeviator = 255
dataset = uds.ToolDataset('/home/bennyg/Development/datasets/instrument_dataset_1', mean, 
                      std, pixdeviator, input_size)
img, mask = dataset[0]

print(img.shape)
print(mask.shape)