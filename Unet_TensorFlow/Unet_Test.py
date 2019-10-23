# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
from Unet import Unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import set_session

def load_image(path):
    img=cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (256, 256))
    return np.array([(img.astype(np.float32) / 255)])
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

model = Unet(3)
model.build(((None, 256, 256, 3)))
model.load_weights('/pretrined_models/UNET_CAT_END_C4.h5')

image = load_image('/datasets/miccai_challenge_2018_release_1/seq_1/left_frames/frame000.png')
pred = model.predict(image)
pred = pred[0]
pred = pred.astype(np.int32)
cv2.imwrite('/datasets/test001.png', pred)

