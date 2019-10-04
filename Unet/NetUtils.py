# -*- coding: utf-8 -*-
import torch
from Unet import UNet

def load_model(protoPath, numclasses):
    net = UNet(numclasses)
    net.load_state_dict(torch.load(protoPath))
    net.eval()
    return net

def save_model(model, protoPath):
    torch.save(model.state_dict(), protoPath)
