# -*- coding: utf-8 -*-
import tensorflow as tf
from Dataset import  MiccaiDataset
from Unet import Unet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
                       '/datasets/miccai_challenge_release_4/seq_16'], 255, (256, 256))

image_srcs, image_labels = data.load_images()

model = Unet(3)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(image_srcs.shape)
print('Generating images')
datagen = ImageDataGenerator()
train_data = datagen.flow(image_srcs, image_labels, batch_size=batchSize)
#train_labels = datagen.flow(image_labels, batch_size=batchSize)
print('image generation done')

model.fit(train_data, epochs=15)


