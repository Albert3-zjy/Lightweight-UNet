# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:08:14 2022

@author: DELL
"""
import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_shape, num_classes):
    image_size1, image_size2 = image_shape[0], image_shape[1]
    model_input = keras.Input(shape=(image_size1, image_size2, 3))
    """ ResNet50 """
    resnet50 = keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=model_input
    )
    # 获取所有层，返回层对象列表
    layers_l = resnet50.layers
    # 获取每一层名字 #
    for layer in layers_l:
        print(layer.name)
    # End #
    x = resnet50.get_layer("conv4_block6_2_relu").output
    """ ResNet101 """
    # resnet101 = keras.applications.ResNet101(
        # weights=None, include_top=False, input_tensor=model_input
    # )
    # # 获取所有层，返回层对象列表
    # layers_l = resnet101.layers
    # # 获取每一层名字 #
    # for layer in layers_l:
        # print(layer.name)
    # # End #
    # x = resnet101.get_layer("conv4_block23_2_relu").output
    """ MobileNet-V2 """
    # mobilenetv2 = keras.applications.mobilenet_v2.MobileNetV2(weights=None, include_top=False, input_tensor=model_input)
    # # 获取所有层，返回层对象列表
    # layers_l = mobilenetv2.layers
    # # 获取每一层名字 #
    # for layer in layers_l:
        # print(layer.name)
    # # End #
    # x = mobilenetv2.get_layer("block_16_depthwise_relu").output
    """ Xception """
    # xception_model = keras.applications.xception.Xception(include_top=False, weights=None, input_tensor=model_input)
    # # 获取所有层，返回层对象列表
    # layers_l = xception_model.layers
    # # 获取每一层名字 #
    # for layer in layers_l:
    #     print(layer.name)
    # # End #
    # x = xception_model.get_layer("block13_sepconv2_act").output
    """ End """
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size1 // 4 // x.shape[1], image_size2 // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    """ ResNet50 """
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    """ ResNet101 """
    # input_b = resnet101.get_layer("conv2_block3_2_relu").output
    """ MobileNet-V2 """
    # input_b = mobilenetv2.get_layer("block_2_depthwise_relu").output
    """ Xception """
    # input_b = xception_model.get_layer("block4_sepconv2_act").output
    # input_b = layers.UpSampling2D(
    #     size=(2, 2),interpolation="bilinear")(input_b)
    """ End """
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size1 // x.shape[1], image_size2 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    ### Additions: activation layer ###
    model_output = tf.keras.layers.Activation('sigmoid')(model_output)
    ### End ###
    return keras.Model(inputs=model_input, outputs=model_output)


