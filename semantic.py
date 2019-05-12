from dataset import load_image
import torch
from generate_masks import get_model

model_path = 'd:\/Development/model_0.pt'
model = get_model(model_path, model_type='LinkNet34', problem_type='binary')