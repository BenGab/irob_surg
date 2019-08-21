from dataset import load_image
import torch
from generate_masks import get_model
from functions import img_to_tensor
import numpy as np

arrt = np.random.randn(500, 500)
tensor = img_to_tensor(arrt)
print(tensor)

model_path = 'pretrained/linknet_binary_20/model_0.pt'
model = get_model(model_path, model_type='LinkNet34', problem_type='binary')