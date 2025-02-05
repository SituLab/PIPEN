# -*- coding: utf-8 -*-

import os
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

from PIL import Image
from scipy import signal
# from get_train import Phy_Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Convolution2D, LeakyReLU, BatchNormalization, UpSampling2D, Dropout, Activation, Flatten, \
    Dense, Lambda, Reshape, concatenate, Convolution3D,UpSampling3D,Add,MaxPooling2D
from skimage.metrics import structural_similarity as SSIM
import h5py
import time as t
import scipy.io as scio
from sklearn.metrics import mean_squared_error
# from skimage.measure import compare_mse
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import gc


def build_unet():
    """
    Create the U-Net Generator using the hyperparameter values defined below
    """
    kernel_size = 5
    strides = 2
    leakyrelu_alpha = 0.2
    upsampling_size = (2, 2)
    dropout = 0.5
    output_channels = 96
    output = (1, 48, 48)
    input_shape = (96, 96, 2)

    input_layer = Input(shape=input_shape)

    # Encoder Network
    enc1 = Convolution2D(filters=32,kernel_size=kernel_size,strides=(2,2),padding='same',name='input2')(input_layer)
    # 1st Convolutional block in the encoder network
    encoder1 = Convolution2D(filters=64, kernel_size=kernel_size, padding='same',
                             strides=strides,name='enc1')(enc1)
    encoder1 = LeakyReLU(alpha=leakyrelu_alpha,name='enl1')(encoder1)
    # encoder1 = Activation('relu')(encoder1)

    # 2nd Convolutional block in the encoder network
    encoder2 = Convolution2D(filters=128, kernel_size=kernel_size, padding='same',
                             strides=strides,name='enc2')(encoder1)
    encoder2 = BatchNormalization(name='enb2')(encoder2)
    encoder2 = LeakyReLU(alpha=leakyrelu_alpha,name='enl2')(encoder2)
    # encoder2 = Activation('relu')(encoder2)

    # 3rd Convolutional block in the encoder network
    encoder3 = Convolution2D(filters=256, kernel_size=kernel_size, padding='same',
                             strides=strides,name='enc3')(encoder2)
    encoder3 = BatchNormalization(name='enb3')(encoder3)
    encoder3 = LeakyReLU(alpha=leakyrelu_alpha,name='enl3')(encoder3)
    # encoder3 = Activation('relu')(encoder3)

    # 4th Convolutional block in the encoder network
    encoder4 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same',
                             strides=strides,name='enc4')(encoder3)
    encoder4 = BatchNormalization(name='enb4')(encoder4)
    encoder4 = LeakyReLU(alpha=leakyrelu_alpha,name='enl4')(encoder4)
    # encoder4 = Activation('relu')(encoder4)

    # # 5th Convolutional block in the encoder network
    # encoder5 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same',
    #                          strides=strides)(encoder4)
    # encoder5 = BatchNormalization()(encoder5)
    # # encoder5 = LeakyReLU(alpha=leakyrelu_alpha)(encoder5)
    # encoder5 = Activation('relu')(encoder5)

    # Decoder Network

    # 1st Upsampling Convolutional Block in the decoder network
    decoder1 = UpSampling2D(size=upsampling_size,name='deu1')(encoder4)
    decoder1 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same',name='dec1')(decoder1)
    decoder1 = BatchNormalization(name='deb1')(decoder1)
    decoder1 = Dropout(dropout,name='ded1')(decoder1)
    decoder1 = concatenate([decoder1, encoder3], axis=3,name='decc1')
    decoder1 = LeakyReLU(alpha=leakyrelu_alpha,name='del1')(decoder1)

    # 2nd Upsampling Convolutional block in the decoder network
    decoder2 = UpSampling2D(size=upsampling_size,name='deu2')(decoder1)
    decoder2 = Convolution2D(filters=256, kernel_size=kernel_size, padding='same',name='dec2')(decoder2)
    decoder2 = BatchNormalization(name='deb2')(decoder2)
    decoder2 = Dropout(dropout,name='ded2')(decoder2)
    decoder2 = concatenate([decoder2, encoder2], axis=3,name='decc2')
    decoder2 = LeakyReLU(alpha=leakyrelu_alpha,name='del2')(decoder2)

    # 3rd Upsampling Convolutional block in the decoder network
    decoder3 = UpSampling2D(size=upsampling_size,name='deu3')(decoder2)
    decoder3 = Convolution2D(filters=128, kernel_size=kernel_size, padding='same',name='dec3')(decoder3)
    decoder3 = BatchNormalization(name='deb3')(decoder3)
    decoder3 = Dropout(dropout,name='ded3')(decoder3)
    decoder3 = concatenate([decoder3, encoder1], axis=3,name='decc3')
    decoder3 = LeakyReLU(alpha=leakyrelu_alpha,name='del3')(decoder3)

    # 4th Upsampling Convolutional block in the decoder network
    decoder4 = UpSampling2D(size=upsampling_size,name='deu4')(decoder3)
    decoder4 = Convolution2D(filters=64, kernel_size=kernel_size, padding='same',name='dec4')(decoder4)
    decoder4 = BatchNormalization(name='deb4')(decoder4)
    # decoder4 = concatenate([decoder4, encoder1], axis=3)
    decoder4 = LeakyReLU(alpha=leakyrelu_alpha,name='del4')(decoder4)

    # Last Convolutional layer
    decoder5 = UpSampling2D(size=upsampling_size)(decoder4)
    decoder5 = Convolution2D(filters=5, kernel_size=kernel_size, padding='same',name='dec5')(decoder5)
    decoder5 = BatchNormalization(name='deb5')(decoder5)
    # decoder5 = Activation('sigmoid')(decoder5)
    decoder5 = Activation('relu',name='output')(decoder5)

    model = Model(inputs=[input_layer], outputs=[decoder5])
    return model

def ResNet():
    input_shape = (96, 96, 2)

    input_layer = Input(shape=input_shape)
    v1 = Convolution2D(filters=64, kernel_size=(5, 5), padding='same')(input_layer)
    v1 = LeakyReLU(alpha=0.3)(v1)
    v1 = BatchNormalization()(v1)

    v2 = MaxPooling2D(pool_size=(2, 2))(v1)

    v3 = res_block1(v2)
    v3 = concatenate([v2, v3])
    v3 = LeakyReLU(alpha=0.3)(v3)


    v11 = Flatten()(v3)
    v11 = Dense(units=1600)(v11)
    v11 = LeakyReLU(alpha=0.3)(v11)

    v12 = Dense(units=73728)(v11)
    v12 = Activation('relu')(v12)

    v13 = Reshape((96, 96, 8))(v12)

    model = Model(inputs=[input_layer], outputs=[v13])
    return model


def res_block1(input):
    b1 = Convolution2D(filters=32, kernel_size=(1, 1), padding='same')(input)
    b1 = LeakyReLU(alpha=0.3)(b1)
    b2 = Convolution2D(filters=32, kernel_size=(3, 3), padding='same')(b1)
    b2 = LeakyReLU(alpha=0.3)(b2)
    b3 = Convolution2D(filters=32, kernel_size=(1, 1), padding='same')(b2)
    return b3


def res_block2(input):
    b1 = Convolution2D(filters=64, kernel_size=(1, 1), padding='same')(input)
    b1 = LeakyReLU(alpha=0.3)(b1)
    b2 = Convolution2D(filters=64, kernel_size=(3, 3), padding='same')(b1)
    b2 = LeakyReLU(alpha=0.3)(b2)
    b3 = Convolution2D(filters=64, kernel_size=(1, 1), padding='same')(b2)
    return b3