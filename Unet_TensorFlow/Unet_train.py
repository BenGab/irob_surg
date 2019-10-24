# -*- coding: utf-8 -*-
import tensorflow as tf
from Dataset import  MiccaiDataset
from Unet import Unet, Unet11
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import set_session
from time import time
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

best_save_path='/pretrined_models/UNET11_MAE_C1_best.h5'
save_path = '/pretrined_models/UNET11_MAE_C1_.h5'
callbacks_l = [
    TensorBoard('/boards/{}'.format(time())),
    EarlyStopping('loss', patience=10, verbose=1),
    ReduceLROnPlateau('loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1),
    ModelCheckpoint(best_save_path, 'loss' ,verbose=1, save_best_only=True, save_weights_only=True)
]
batchSize = 1
data = MiccaiDataset(['/datasets/miccai_challenge_2018_release_1/seq_1'], 255, (256, 256), batch_size=batchSize)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

train_images = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=2, height_shift_range=2)
mask_images = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=2, height_shift_range=2)
seed = 1
s_im, s_mask = data.data_sample()
print(s_mask.shape)
train_images.fit([s_im], augment=True, seed=seed)
mask_images.fit([s_mask], augment=True, seed=seed)

image_generator = train_images.flow_from_directory('/datasets/miccai/images/data/train', target_size=(256, 256), batch_size=batchSize, class_mode=None, seed=seed, shuffle=False)
mask_generator = mask_images.flow_from_directory('/datasets/miccai/labels/data/train', target_size=(256, 256), batch_size=batchSize, class_mode=None, seed=seed, shuffle=False)

generator = (image_generator, mask_generator)
net = Unet11()
model = net.build_unet(tf.keras.layers.Input((256, 256, 3)))
model.compile(loss='mean_absolute_error', optimizer=Adam(0.01), metrics=['accuracy'])

model.fit(image_generator, mask_generator, epochs=100, callbacks=callbacks_l)
model.save_weights(save_path)



