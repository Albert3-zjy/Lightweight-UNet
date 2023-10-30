# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:22:44 2018

@author: JXDUAN
"""

"""
U-Net model
"""
"""
U-Net based on Keras
"""# Build U-Net model
import os
import sys
import random
import warnings

import numpy as np
# import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, UpSampling2D

from keras import backend as K

import tensorflow as tf

def Conv_block(m, dim, activation, batchnorm, residual, do=0):
    n = Conv2D(dim, 3, activation=activation, padding='same')(m)
    n = BatchNormalization()(n) if batchnorm else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=activation, padding='same')(n)
    n = BatchNormalization()(n) if batchnorm else n
    if residual:
        n = concatenate([m, n])
    return n

def Up_block(m, dim, activation, batchnorm, residual, up):
    if up:
        m = UpSampling2D()(m)
        m = Conv2D(dim, 2, activation=activation, padding='same')(m)
    else:
        m = Conv2DTranspose(dim, 3, strides=2, activation=activation, padding='same')(m)
        m = Conv_block(n, dim, activation, batchnorm, residual)
    return m

def build_model(shape, num=1):
    
    inputs = Input(shape)
    s = Lambda(lambda x: x / 255) (inputs)
    ch = 16     # Origin: ch = 16

    c1 = Conv2D(ch, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(ch, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = Conv_block(c1, ch, activation='relu', batchnorm=True, residual=True)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(ch*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(ch*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = Conv_block(c2, ch*2, activation='relu', batchnorm=True, residual=True, do=0)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(ch*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(ch*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = Conv_block(c3, ch*4, activation='relu', batchnorm=True, residual=True, do=0)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(ch*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(ch*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    c4 = Conv_block(c4, ch*8, activation='relu', batchnorm=True, residual=True, do=0)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(ch*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.2) (c5)
    c5 = Conv2D(ch*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
    c5 = Conv_block(c5, ch*16, activation='relu', batchnorm=True, residual=True, do=0)
    p5 = MaxPooling2D(pool_size=(2, 2)) (c5)

    c6 = Conv2D(ch*32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p5)
    c6 = Dropout(0.3) (c6)
    c6 = Conv2D(ch*32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(ch*16, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(ch*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(ch*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(ch*8, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(ch*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.2) (c8)
    c8 = Conv2D(ch*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(ch*4, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(ch*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.2) (c9)
    c9 = Conv2D(ch*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    u10 = Conv2DTranspose(ch*2, (2, 2), strides=(2, 2), padding='same') (c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(ch*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u10)
    c10 = Dropout(0.1) (c10)
    c10 = Conv2D(ch*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c10)

    u11 = Conv2DTranspose(ch, (2, 2), strides=(2, 2), padding='same') (c10)
    u11 = concatenate([u11, c1], axis=3)
    c11 = Conv2D(ch, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u11)
    c11 = Dropout(0.1) (c11)
    c11 = Conv2D(ch, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c11)

    outputs = Conv2D(num, (1, 1), activation='sigmoid') (c11)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

