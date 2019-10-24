# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
from Unet import Unet, Unet11
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import set_session

def load_image(path):
    img=cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (256, 256))
    return np.array([(img.astype(np.float32) / 255)])
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

net = Unet11()
input_img = tf.keras.layers.Input((256, 256, 3), name='img')
model = net.build_unet(input_img)
model.load_weights('/pretrined_models/UNET11_CAT_END_C3_best.h5')

image = load_image('/datasets/miccai_challenge_2018_release_1/seq_1/left_frames/frame000.png')
pred = model.predict(image)
pred = pred[0]
pred = pred * 255
pred = pred.astype(np.int32)
pred[pred < 0]=0
pred[pred > 255]=0
cv2.imwrite('/datasets/UNET11_CAT_END_C3_best.png', pred)

