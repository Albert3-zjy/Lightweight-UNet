# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:57:50 2022

@author: DELL
"""
import os
import glob
import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt


def color_label_trans(imapath,labelpath,savepath,num_classes):
    
    "颜色：['b','g','r'] format"
    kolor_value_bar = [[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.803,0.522,0.247],[0.518,0.439,1]]

    ima = cv2.imread(imapath)
    im_label = cv2.imread(labelpath)
    if len(im_label.shape) == 3:
        im_label = cv2.cvtColor(im_label, cv2.COLOR_RGB2GRAY)
        
    # io.imshow(im_label)
    # plt.show()
    
    H, W = im_label.shape
    
    # imm = np.zeros((H,W,3))
    imm = ima
    for cc in range(1,num_classes):
        Kolor_value = kolor_value_bar[cc]
        imm[im_label==cc,0] = Kolor_value[0]*255
        imm[im_label==cc,1] = Kolor_value[1]*255
        imm[im_label==cc,2] = Kolor_value[2]*255
    
    # io.imshow(imm)
    # plt.show()
    
    cv2.imwrite(savepath,imm)


def label_machen(imapath,savepath):
    im_label = cv2.imread(imapath)
    if len(im_label.shape) == 3:
        im_label = cv2.cvtColor(im_label, cv2.COLOR_RGB2GRAY)
        
    # io.imshow(im_label)
    # plt.show()
    
    H, W = im_label.shape
    
    im_label_neu = np.zeros((H,W))
    
    # 2nd class: Palatal, 3rd class: ena
    im_label_neu[im_label==2] = 1
    im_label_neu[im_label==3] = 2
    
    # io.imshow(imm)
    # plt.show()
    
    cv2.imwrite(savepath,im_label_neu)
    

def im_crop(im,crop_win):
    H_st, H_end = crop_win[0,0], crop_win[0,1]
    W_st, W_end = crop_win[1,0], crop_win[1,1]
    im_neu = im[H_st:H_end, W_st:W_end, :]
    
    return im_neu   


# annotations = glob.glob('./data_palatal/*_label.png')
# annotations = glob.glob('D:/DQYJY/teeth/DATA/2labels/train_crop/*label.png')   #段博
# images = glob.glob('D:/DQYJY/teeth/DATA/2labels/train_crop/*image.png')
# savedir = 'D:/DQYJY/teeth/DATA/2labels/train_biaozhu_crop/'
annotations = glob.glob('../DATA/2labels/test_neu/*label.png')    #海雯
images = glob.glob('../DATA/2labels/test_neu/*image.png')
savedir = '../DATA/2labels/test_neu_crop'

num_classes = 3

for ii in range(len(annotations)):
    labelpath = annotations[ii]
    imapath = images[ii]
    # savepath = imapath[:-4]+'_c.png'
    
    """ 从人工标注中类别显示 """
    # _, fname = os.path.split(imapath)
    # savepath = os.path.join(savedir,fname)
    # color_label_trans(imapath,labelpath,savepath,num_classes)
    
    """ 从人工标注中选取类别 """
    # _, fname = os.path.split(labelpath)
    # savepath = os.path.join(savedir,fname)
    # label_machen(labelpath,savepath)
    
    """ 裁剪 """
    _, fname_im = os.path.split(imapath)
    _, fname_la = os.path.split(labelpath)
    savepath_im = os.path.join(savedir,fname_im)
    savepath_la = os.path.join(savedir,fname_la)
    img = cv2.imread(imapath)
    label = cv2.imread(labelpath)
    H_st, W_st = 140, 330
    H_len, W_len = 320, 384
    crop_win = np.array([[H_st, H_st+H_len],[W_st, W_st+W_len]])
    img_crop, label_crop = im_crop(img,crop_win), im_crop(label,crop_win)
    cv2.imwrite(savepath_im,img_crop)
    cv2.imwrite(savepath_la,label_crop)
    
print("\nSaving done!")

