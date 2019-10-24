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
data = MiccaiDataset(['/datasets/miccai_challenge_2018_release_1/seq_1',
                       '/datasets/miccai_challenge_2018_release_1/seq_2',
                       '/datasets/miccai_challenge_2018_release_1/seq_3',
                       '/datasets/miccai_challenge_2018_release_1/seq_4',
                       '/datasets/miccai_challenge_release_2/seq_5',
                       '/datasets/miccai_challenge_release_2/seq_6',
                       '/datasets/miccai_challenge_release_2/seq_7',
                       '/datasets/miccai_challenge_release_3/seq_9',
                       '/datasets/miccai_challenge_release_3/seq_10',
                       '/datasets/miccai_challenge_release_3/seq_11',
                       '/datasets/miccai_challenge_release_3/seq_12'
                       '/datasets/miccai_challenge_release_4/seq_13', 
                       '/datasets/miccai_challenge_release_4/seq_14',
                       '/datasets/miccai_challenge_release_4/seq_15',
                       '/datasets/miccai_challenge_release_4/seq_16'], 255, (256, 256), batch_size=batchSize)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

net = Unet11()
model = net.build_unet(tf.keras.layers.Input((256, 256, 3)))
model.compile(loss='mean_absolute_error', optimizer=Adam(0.01), metrics=['accuracy'])
datagen = data.image_generator()

model.fit_generator(datagen, epochs=100, steps_per_epoch=data.data_len(), callbacks=callbacks_l)
model.save_weights(save_path)



