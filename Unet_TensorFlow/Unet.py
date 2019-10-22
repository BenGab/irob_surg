# -*- coding: utf-8 -*-
import tensorflow as tf

def double_conv(out_channels):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(out_channels, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(out_channels, 3, padding='same', activation='relu')
    ])

class Unet(tf.keras.Model):
    def __init__(self, num_classes):
        super(Unet, self).__init__()
        self.num_classes = num_classes
        self.conv_layer_1 = double_conv(64)
        self.conv_layer_2 = double_conv(128)
        self.conv_layer_3 = double_conv(256)
        self.conv_layer_4 = double_conv(512)
        self.maxpool = tf.keras.layers.MaxPool2D()
        self.upsample = tf.keras.layers.UpSampling2D(interpolation='bilinear')
        self.concatenate = tf.keras.layers.Concatenate()
        self.out_conv = tf.keras.layers.Conv2D(num_classes, 1, activation='relu')

        self.upconv_layer_3 = double_conv(256)
        self.upconv_layer_2 = double_conv(128)
        self.upconv_layer_1 = double_conv(64)
    
    def call(self, x):
        print(x)
        conv1 = self.conv_layer_1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv_layer_2(x)
        x = self.maxpool(conv2)

        conv3 = self.conv_layer_3(x)
        x = self.maxpool(conv3)

        x = self.conv_layer_4(x)

        x = self.upsample(x)
        x = self.concatenate([x, conv3])

        x = self.upconv_layer_3(x)
        
        x = self.upsample(x)
        x = self.concatenate([x, conv2])

        x = self.upconv_layer_2(x)

        x = self.upsample(x)
        x = self.concatenate([x, conv1])

        x = self.upconv_layer_1(x)
        x = self.out_conv(x)
        return x