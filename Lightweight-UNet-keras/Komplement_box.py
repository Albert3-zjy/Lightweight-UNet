# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:10:31 2022

@author: DELL
"""
import random, math, os
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import json
import glob
import cv2
from skimage import io
import math


def find_edgePoints(elements, k, b_new, f_cof):
        distance = abs(k * elements[0] + b_new - elements[1]) / np.sqrt(k ** 2 + 1)
        # print(distance.min())
        index = np.where(distance < 0.5)
        X = elements[0][index]
        Y = elements[1][index]
        if len(X)==0:
            return 'nan'
        X_1, X_2 = X.min(), X.max()
        Y_1, Y_2 = k * X_1 + b_new, k * X_2 + b_new

        return X_2, Y_2


def intersection(start1, end1, start2, end2):
    def getLine(start, end):
        x1, y1 = start
        x2, y2 = end
        return y2 - y1, x1 - x2, x2 * y1 - x1 * y2
 
    def inLine(x, y):
        for s, e in ((start1, end1), (start2, end2)):
            minx = min(s[0], e[0])
            miny = min(s[1], e[1])
            maxx = max(s[0], e[0])
            maxy = max(s[1], e[1])
            if not (minx <= x <= maxx and miny <= y <= maxy):
                return False
        return True
 
    a1, b1, c1 = getLine(start1, end1)
    a2, b2, c2 = getLine(start2, end2)
 
    if a1 * b2 == a2 * b1: # 平行
        if c1 != c2: # 平行但不共线
            return []
        # 共线
        res = []
        for p in (start1, end1, start2, end2):
            if inLine(p[0], p[1]): # 四个点有任何一个在线段上
                if not res or p[0] < res[0] or p[0] == res[0] and p[1] < res[1]:
                    res = p
        return res
 
    # 不平行，计算交点
    x = (b1 * c2 - c1 * b2) / (b2 * a1 - a2 * b1)
    y = (a1 * c2 - a2 * c1) / (a2 * b1 - a1 * b2)

    # if inLine(x, y):
    #     return [x, y]
    # else:
    #     return []
    return [x, y]

def Komplement(elements_target,elements_eCe,point_eCe, point_youzhi_left, point_youzhi_right, nHeight, f_cof, disp=False):

    elements = elements_target
    x1, y1 = point_youzhi_left
    x2, y2 = point_youzhi_right
    # x1, y1 = 471.2, 319.4
    # x2, y2 = 498.4, 300.7
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k*x1
    cos_k = (x2 - x1) / np.sqrt((x2-x1)**2 + (y2-y1)**2)
    distance = abs(k*elements[0] + b - elements[1]) / np.sqrt(k**2 + 1)
    distance_min = distance.min()
    distance_max = distance.max()
    # print("\ndistance_min: {}, distance_max: {}".format(distance_min* f_cof,distance_max* f_cof))
    
    # apical点 #
    x_0, y_0 = point_eCe  # 腭侧区域上方第一个点为根尖点，也是延长线上的start点：[x3, y3]
    b_0 = y_0 - k*x_0
    # End #
    
    # f_cof = 0.1976227  # 1个像素点为0.197mm
    dis_n = 2 # 每2毫米测一次骨璧厚度
    n1, n2 = np.ceil(distance_min * f_cof / dis_n) , np.ceil(distance_max * f_cof / dis_n)

    # dis_out = []
    element_edge = []

    # 计算额侧骨璧厚度，从下（2点线）每隔2mm向上
    for i in range(int(n1), int(n2)):
        h = dis_n/f_cof * i
        b_new = b + h / cos_k
        try:
            X_2, Y_2 = find_edgePoints(elements, k, b_new, f_cof)
        except:
            continue

        element_edge.append([int(X_2), int(Y_2)])
    
    # 计算额侧骨璧厚度，从上点（2点线在1点-即腭侧骨璧左侧点）每隔2mm向下
    # for i in range(int(n2)):
    #     h = dis_n / f_cof * i
    #     b_new = b_0 - h / cos_k
    #     try:
    #         X_2, Y_2 = find_edgePoints(elements, k, b_new, f_cof)
    #     except:
    #         continue

    #     element_edge.append([int(X_2), int(Y_2)])

    ### 腭侧区域右侧边缘点拟合曲线 ###
    # print("\nelement_edge shape: ",np.shape(np.array(element_edge)))
    
    element_edge_arr = np.array(element_edge)

    # cej根尖点向上 #
    x_list, y_list = element_edge_arr[:-2,0], element_edge_arr[:-2,1]
    # apical根尖点向下 #
    # x_list, y_list = element_edge_arr[1:,0], element_edge_arr[1:,1]
    # End #
    if len(x_list) != 0:
        # print("\nlen x_list: ",len(x_list))
        order = 1
        coefs = poly.polyfit(x_list, y_list, order)
        fit_func = poly.Polynomial(coefs)        # 拟合曲线函数
        ### End ###
        ### 过apical点曲线延伸线上的一点 ###
    
        i = 0
        h = dis_n / f_cof * i
        b_new = b_0 - h / cos_k
        x_neu = int(elements_target[0].max())
        y_neu = k*x_neu + b_new       # 延长线上的end点：[x_neu,y_neu]
        
        x_list = x_list.tolist()
        x_list.append(int(x_neu))
        y_neu_fit = fit_func(np.array(x_list))   # 拟合线上的end点：[x_neu,y_neu_fit] 
        # plt.plot(x_neu, y_neu, )
        
        if disp:
            plt.plot(np.array(x_list), nHeight-np.array(y_neu_fit),'y-')
        
        x_edge0, y_edge0 = element_edge[0][0], element_edge[1][0]      # 拟合线上的start点：[x_edge, y_edge] 
        
        ### 延长线与拟合线交叉点 ###
        start1, end1 = [x_0, y_0], [x_neu, y_neu]                 # 延长线
        start2, end2 = [x_edge0, y_edge0], [x_neu, y_neu_fit[-1]]     # 拟合线
        [x_inter, y_inter] = intersection(start1, end1, start2, end2)
        Komp = 0         # 沿延长线向外延伸几个像素
        x_inter_neu = x_inter + Komp
        y_inter_neu = k*x_inter_neu + b_0
        thickness_neu = np.sqrt((x_inter - x_0) ** 2 + (y_inter - y_0) ** 2)
    else:
        thickness_neu = np.nan
        x_inter, y_inter = np.nan, np.nan
    return thickness_neu*f_cof, [x_inter_neu, y_inter_neu]
    
