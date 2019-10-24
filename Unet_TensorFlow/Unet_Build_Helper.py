import tensorflow as tf
from tensorflow.keras.backend import set_session
from Unet import Unet11, Unet11_G

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

input_img = tf.keras.layers.Input((256, 256, 1), name='img')
net = Unet11_G()

model = net.build_unet(input_img)
model.summary()