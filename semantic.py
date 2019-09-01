from dataset import load_image
import torch
from generate_masks import get_model, img_transform
from functions import img_to_tensor
import numpy as np
import cv2

arrt = np.random.randn(500, 500)
tensor = img_to_tensor(arrt)
print(tensor)

model_path = 'pretrained/linknet_binary_20/model_0.pt'
model = get_model(model_path, model_type='LinkNet34', problem_type='binary')

img =  load_image('pretrained/test2.png')
img_tensor = img_to_tensor(img)
input_img = torch.unsqueeze(img_to_tensor(img_transform(p=1)(image=img)['image']), dim=0)
mask = model(input_img)
print("printing mask")
print(mask)