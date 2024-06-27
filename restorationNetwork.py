import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Flatten, Dense, concatenate, Reshape, UpSampling2D, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, GlobalAveragePooling2D, MaxPooling2D, Concatenate
from keras.models import load_model
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from dataGeneration import *
from tensorflow.image import psnr

# Defining convolution block
'''
Function for a single convolution block
[CONV -> BN -> RELU] -> [CONV -> BN -> RELU]
Params:
    input: input tensor
    n_filters: number of filters
'''
def convBlock(input, n_filters):
    x = Conv2D(n_filters, (3, 3), padding='same',kernel_initializer=tf.keras.initializers.HeNormal(0), use_bias=False)(input)
    x = BatchNormalization()(x) 
    x = Activation('gelu')(x)

    x = Conv2D(n_filters, (3, 3), padding='same',kernel_initializer=tf.keras.initializers.HeNormal(0), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)

    x = Conv2D(n_filters, (3, 3), padding='same',kernel_initializer=tf.keras.initializers.HeNormal(0), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)

    x = Conv2D(n_filters, (3, 3), padding='same',kernel_initializer=tf.keras.initializers.HeNormal(0), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)

    x = Conv2D(n_filters, (3, 3), padding='same',kernel_initializer=tf.keras.initializers.HeNormal(0), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)

    return x

'''
Function for a single encoder layer
Params:
    input: input tensor
    n_filters: total number of filters
'''
def encoderLayer(input, n_filters):
    x = convBlock(input, n_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

'''
Function for a single decoder layer
Params:
    input: input tensor
    skip_connection: skip connection tensor
    n_filters: total number of filters
'''
def decoderLayer(input, skip_connections, n_filters):
    x = Conv2DTranspose(n_filters, (2, 2), strides=2, padding='same')(input)
    x = Concatenate()([x, skip_connections])
    x = convBlock(x, n_filters)
    return x

'''
'''
def generateUNet():

    # Setting input
    inputs = Input(shape=(128, 128, 1))

    # Encoder layers
    s1, p1 = encoderLayer(inputs, 16)
    s2, p2 = encoderLayer(p1, 32)
    s3, p3 = encoderLayer(p2, 64)
    s4, p4 = encoderLayer(p3, 128)

    # Bridge Layer
    b1 = convBlock(p4, 256)

    # Decoder path
    d1 = decoderLayer(b1, s4, 128)
    d2 = decoderLayer(d1, s3, 64)
    d3 = decoderLayer(d2, s2, 32)
    d4 = decoderLayer(d3, s1, 16)

    # Output layer
    outputs = Conv2D(1, (1, 1), padding='same', activation='linear', kernel_initializer=tf.keras.initializers.HeNormal(0),use_bias=False)(d4)
    outputs = inputs + outputs
    model = Model(inputs, outputs)
    return model


def PSNRMetric(y_true, y_pred):
    return psnr(y_true, y_pred, max_val=1.0)

def SSIMMetric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)