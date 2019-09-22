from dataset import load_image
import torch
from generate_masks import get_model, img_transform
from functions import img_to_tensor
import numpy as np
import matplotlib.pyplot as plt

arrt = np.random.randn(500, 500)
tensor = img_to_tensor(arrt)
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

#new_img = img - mean /std
#old img = 

model_path = 'pretrained/unet16_instruments_20/model_1.pt'
model = get_model(model_path, model_type='UNet16', problem_type='instruments')

img =  load_image('dataset/instrument_dataset_1/left_frames/frame000.png')
img_tensor = img_to_tensor(img)
input_img = torch.unsqueeze(img_to_tensor(img_transform(p=1)(image=img)['image']), dim=0)
mask = model(input_img)

im_seg = mask.data[0].cpu().numpy()[0]

#mask_array = (im_seg * std) + mean

plt.imshow(im_seg > 0)
