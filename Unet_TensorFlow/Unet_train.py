# -*- coding: utf-8 -*-
import tensorflow as tf
from Dataset import  MiccaiDataset
from Unet import Unet, Unet11
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import set_session
from time import time
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

best_save_path='/pretrined_models/UNET11_MSE_C2_best.h5'
save_path = '/pretrined_models/UNET11_MSE_C2.h5'
callbacks_l = [
    TensorBoard('/boards/UNET11_MSE_C2'),
    ReduceLROnPlateau('loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1),
    ModelCheckpoint(best_save_path, verbose=1, save_best_only=True, save_weights_only=True)
]
batchSize = 1
data = MiccaiDataset('/datasets/miccai/dataset', 255, (256, 256), batch_size=batchSize)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

net = Unet11()
model = net.build_unet(tf.keras.layers.Input((256, 256, 3)))
model.compile(loss='mean_squared_error', optimizer=Adam(0.01), metrics=['accuracy'])
datagen_train = data.image_train_generator()
datagen_val = data.image_val_generator() 

model.fit_generator(datagen_train, epochs=20, steps_per_epoch=data.data_train_len(), callbacks=callbacks_l,
                    validation_data=datagen_val, validation_steps=data.data_val_len())
model.save_weights(save_path)



