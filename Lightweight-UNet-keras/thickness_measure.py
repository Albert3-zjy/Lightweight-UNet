# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:08:15 2022

@author: DELL
"""
import random, math, os
import numpy as np
import pandas as pd
import xlrd, xlwt
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import json
import glob
import cv2
from skimage import io
from skimage.metrics import mean_squared_error as compare_mse
from Komplement_box import Komplement
import math


def get_point_eCe(nHeight, nWidth, elements_eCe, coordinate_eCe, k=1, numb_pixel=1):
    # 计算腭侧-根尖点坐标： 用斜率为K的直线（kx+b=y)过额侧区域中心点，确定出b。逐步把曲线一个像素往上移动，最终在有10个像素在曲线上面确定出这10个像素的中心位置——为根尖点
    x0, y0 = coordinate_eCe
    b0 = y0 - k * x0
    x, y = elements_eCe
    ### 寻找根尖点 ###
    for i in range(10000):
        b = b0 + i
        a = y - k * x - b
        index = np.where(a > 0)
        index = index[0]
        if len(index) > numb_pixel:
            index_ok = index
            epoch = i
        else:
            if len(index) > 0:
                index_ok = index
                epoch = i
            # print('epoch=%s, epoch-0k=%s' % (i, epoch))
            y_ok, x_ok = y[index_ok], x[index_ok]
            break
    ### Substitution: 凸包顶点 ###
    # im0 = np.zeros((nHeight, nWidth),dtype=np.uint8)
    # img = np.zeros((nHeight, nWidth, 3),dtype=np.uint8)
    # im0[y.astype('uint8'), x.astype('uint8')] = 1
    # img[y.astype('uint8'), x.astype('uint8'), :] = [255,0,0]
    # contours, hierarchy = cv2.findContours(im0, 2, 1)
    # cnt = contours[0]
    # hull = cv2.convexHull(cnt)
    
    # hull_ = hull.squeeze()
    # # print(hull_)
    
    # # io.imshow(im0)
    # # plt.show()

    # # length = len(hull)
    # # for i in range(len(hull)):
    # #     cv2.line(img, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 2)

    # # io.imshow(img)
    # # plt.show()
    
    # ind = np.where(hull_[:,1] == max(hull_[:,1]))
    # y_ok, x_ok = hull_[ind,1][0], min(hull_[ind,0])
    # print("\ny_ok: {} x_ok: {}".format(y_ok, x_ok))
    ### End ###

    # point_eCe = [int(x_ok.mean()), int(y_ok.mean())]
    point_eCe = [x_ok.mean(), y_ok.mean()]
    print('根尖点（腭侧）坐标：', point_eCe[0], point_eCe[1])
    # plot_pics(elements_eCe, k, b0 + epoch, x0, y0)
    return point_eCe

def get_point_youzhi(nHeight, elements_youzhi, coordinate_youzhi, k=0, numb_pixel=1):
    def plot_youzhi(elements_youzhi, coordinate_youzhi):
        plt.plot(elements_youzhi[0], elements_youzhi[1], 'bo')
        plt.plot(coordinate_youzhi[0], coordinate_youzhi[1], 'r+')
        plt.show()

    # plot_youzhi(elements_youzhi, coordinate_youzhi)
    # 计算釉质-左右两点坐标：
    x, y = elements_youzhi
    x_max = max(x)
    x0, y0 = coordinate_youzhi[0] + (x_max - coordinate_youzhi[0]) * 0.5, coordinate_youzhi[1]
    b0 = y0 - k * x0

    # print(max(x))
    left_right = ['left', 'right']
    for item in left_right:
        if item == 'left':
            index = np.where(x <= x0)
        else:
            index = np.where(x > x0)
        x_new, y_new = x[index], y[index]
        for i in range(10000):
            b = b0 + i
            a = y_new - k * x_new - b
            index = np.where(a > 0)[0]
            if len(index) > numb_pixel:
                index_ok = index
                epoch = i
            else:
                if len(index) > 0:
                    index_ok = index
                    epoch = i
                # print('epoch=%s, epoch-0k=%s' % (i, epoch))
                x_ok, y_ok = x_new[index_ok], y_new[index_ok]
                break
        # point = [int(x_ok.mean()), int(y_ok.mean())]
        # point = [x_ok.mean(), y_ok.mean()]
        index_xOK_max = np.where(x_ok==x_ok.max())[0]
        y_ok_max = y_ok[index_xOK_max]
        point = [x_ok.max(), y_ok_max.max()]
        if item == 'left':
            point_youzhi_left = point
            print('釉质左侧（唇侧）坐标：', point[0], point[1])
        else:
            point_youzhi_right = point
            print('釉质右侧（腭侧）坐标：', point[0], point[1])
        # plot_pics((x_new, y_new), k, b0 + epoch, x0, y0)
    return point_youzhi_left, point_youzhi_right


def get_coordinate_elements(img_array, class_numb, label_target, label_youzhi, label_eCe):
    def append_data(x, y, i, a):
        return np.append(x, a), np.append(y, [i] * len(a[0]))

    # 设置变量
    N = class_numb
    a = np.array([])
    for j in range(N):
        x = 'X_%s' % j
        y = 'Y_%s' % j
        locals()[x], locals()[y] = a, a

    # 每个类别的像素点坐标(x=width, y=height)
    nHeight, nWidth = img_array.shape[0], img_array.shape[1]
    for i in range(nHeight):
        if len(img_array.shape) == 2:
            temp = img_array[nHeight - 1 - i,:]  # 修正图像坐标，原来的width(x),从左到右增大，height(y)从上到下增大————>改为：height(y)从下到上增大， width不变
        else:
            temp = img_array[nHeight - 1 - i, :, 0]
        for j in range(N):
            x = 'X_%s' % j
            y = 'Y_%s' % j
            a = np.where(temp == j)
            locals()[x], locals()[y] = append_data(eval(x), eval(y), i, a)
    # 每个类别的中心点坐标
    coordinate = []
    elements_target = []
    elements_eCe = []
    elements_youzhi = []
    for i in range(N):
        x = 'X_%s' % i
        y = 'Y_%s' % i
        x, y = eval(x), eval(y)
        if i == label_target:
            elements_target = [x, y]
        if i == label_eCe:
            elements_eCe = [x, y]
        if i == label_youzhi:
            elements_youzhi = [x, y]
        x_mean = round(sum(x) / len(x))
        y_mean = round(sum(y) / len(y))
        coordinate.append((x_mean, y_mean))
    point_eCe = get_point_eCe(nHeight, nWidth, elements_eCe, coordinate[label_eCe])
    point_youzhi_left, point_youzhi_right = get_point_youzhi(nHeight, elements_youzhi, coordinate[label_youzhi])

    return coordinate, elements_target, elements_eCe, elements_youzhi, point_eCe, point_youzhi_left, point_youzhi_right


def compute_thickness(elements, k, b_new, f_cof, color='r+'):
        distance = abs(k * elements[0] + b_new - elements[1]) / np.sqrt(k ** 2 + 1)
        # print(distance.min())
        index = np.where(distance < 0.5)
        X = elements[0][index]
        Y = elements[1][index]
        # plt.plot(X,Y,color)
        # plt.show()
        # print(X, Y)
        if len(X)==0:
            return 'nan'
        X_1, X_2 = X.min(), X.max()
        Y_1, Y_2 = k * X_1 + b_new, k * X_2 + b_new
        # plt.plot(X_1,Y_1,'ro')
        # plt.plot(X_2,Y_2,'go')
        # plt.show()
        thickness = np.sqrt((X_2 - X_1) ** 2 + (Y_2 - Y_1) ** 2)
        return thickness*f_cof

def get_thickness_pointFromDistrict(img_array_pred, img, save_path_meas, f_cof=0.1976227, class_numb=5, label_youzhi=2, label_eCe=3, label_target=3, savefig=False):
    
    nHeight = img_array_pred.shape[0]

    def plot_pics_thickness(elements_target, k, b):
        x_min = int(elements_target[0].min() - 30)
        x_max = int(elements_target[0].max() + 30)
        x = np.array([i for i in range(x_min, x_max)])
        y = k*x + b
        plt.plot(x, nHeight-y)
        # plt.plot(elements_target[0], elements_target[1], 'o')
        # plt.show()

    def plot_youzhi(elements_youzhi, point_youzhi_left, point_youzhi_right):
        plt.plot(elements_youzhi[0], elements_youzhi[1], 'bo')
        plt.plot(point_youzhi_left[0], point_youzhi_left[1], 'r+')
        plt.plot(point_youzhi_right[0], point_youzhi_right[1], 'r+')
        plt.show()

    def plot_eCe(elements_eCe, point_eCe):
        plt.plot(elements_eCe[0], elements_eCe[1], 'bo')
        plt.plot(point_eCe[0], point_eCe[1], 'r+')
        plt.show()


    coordinate, elements_target, elements_eCe, elements_youzhi, point_eCe, point_youzhi_left, point_youzhi_right = get_coordinate_elements(img_array_pred, class_numb, label_target, label_youzhi, label_eCe)
    # plot_youzhi(elements_youzhi, point_youzhi_left, point_youzhi_right)
    # plot_eCe(elements_eCe, point_eCe)
    elements = elements_target
    plt.plot(elements_target[0], nHeight-elements_target[1], 'bo')
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
    
    
    # f_cof = 0.1976227  # 1个像素点为0.197mm
    dis_n = 2 # 每2毫米测一次骨璧厚度
    n1, n2 = np.ceil(distance_min * f_cof / dis_n) , np.ceil(distance_max * f_cof / dis_n)
    print("\nn1: {}, n2: {}".format(n1,n2))
    thickness_all_cej = []
    dis_out = []
    plot_pics_thickness(elements_target, k, b)
    # 计算额侧骨璧厚度，从下（2点线）每隔2mm向上
    for i in range(int(n1), int(n2)):
        h = dis_n/f_cof * i
        b_new = b + h / cos_k
        # plot_pics_thickness(elements_target, k, b_new)
        thickness = compute_thickness(elements, k, b_new, f_cof, color='r+')
        thickness_all_cej.append(thickness)
        dis_out.append(h*f_cof)
        # plt.show()
        ### Additions ###
        # distance = abs(k * elements[0] + b_new - elements[1]) / np.sqrt(k ** 2 + 1)
        # index = np.where(distance < 1.5)
        # X = elements[0][index]
        # Y = elements[1][index]
        # plt.plot(X,nHeight-Y,'c-')
        ### End ###
    # plt.show()

    # 计算额侧骨璧厚度，从上点（2点线在1点-即额侧骨璧左侧点）每隔2mm向下
    x_0, y_0 = point_eCe  # 第一个点为根尖点
    # x_0, y_0 = 508.7, 362.5
    b_0 = y_0 - k*x_0
    distance = abs(k * elements[0] + b_0 - elements[1]) / np.sqrt(k ** 2 + 1)
    distance_max = distance.max()
    n2 = np.ceil(distance_max * f_cof / dis_n)
    thickness_all_roof = []
    for i in range(int(n2)):
        h = dis_n / f_cof * i
        b_new = b_0 - h / cos_k
        thickness = compute_thickness(elements, k, b_new, f_cof, color='k*')
        """ 补偿模块 """
        disp_curve = True
        if i == 0:
            thickness_tmp, point_inter = Komplement(elements_target,elements_eCe,point_eCe, point_youzhi_left, point_youzhi_right, nHeight, f_cof, disp=disp_curve)
            if not np.isnan(thickness_tmp):
                thickness = thickness_tmp
            else:
                pass
        """ End """
        thickness_all_roof.append(thickness)
        plot_pics_thickness(elements_target, k, b_new)
        ### Additions ###
        # distance = abs(k * elements[0] + b_new - elements[1]) / np.sqrt(k ** 2 + 1)
        # index = np.where(distance < 1.5)
        # X = elements[0][index]
        # Y = elements[1][index]
        # plt.plot(X,nHeight-Y,'c-')
        ### End ###

    # Display: palatal area #
    x1, y1 = point_youzhi_left
    x2, y2 = point_youzhi_right
    x3, y3 = point_eCe
    x_inter, y_inter = point_inter[0], point_inter[1]

    io.imshow(img)
    plt.plot(x1, nHeight-y1, 'r*')
    plt.plot(x2, nHeight-y2, 'g*')
    plt.plot(x3, nHeight-y3, 'y*')
    plt.plot(x_inter, nHeight-y_inter, 'm*')
    # plt.show()
    if savefig:
        plt.savefig(save_path_meas)
    plt.show()
    # End #

    # print(thickness_all_cej, thickness_all_roof)
    return thickness_all_cej, thickness_all_roof, point_eCe, point_youzhi_left, point_youzhi_right

def get_points_from_jsonFile(filename):
    def deal_decimal(data):
        data[0] = round(data[0] * 10) / 10
        data[1] = round(data[1] * 10) / 10
        return data

    point_eCe_org, point_left_org, point_right_org = [], [], []
    with open(filename, 'rb') as f:
        a = json.load(f)
    imageHeight = a['imageHeight'] - 1
    for i in range(9):
        if a['shapes'][i]['label'] == 'apical':
            point_eCe_org = a['shapes'][i]['points'][0]
            point_eCe_org = deal_decimal(point_eCe_org)
        if a['shapes'][i]['label'] == 'cej_buccal':
            point_left_org = a['shapes'][i]['points'][0]
            point_left_org = deal_decimal(point_left_org)
        if a['shapes'][i]['label'] == 'cej_palatal':
            point_right_org = a['shapes'][i]['points'][0]
            point_right_org = deal_decimal(point_right_org)
    if len(point_eCe_org) > 1 :
        point_eCe_org[-1] = imageHeight - point_eCe_org[-1]
        point_left_org[-1] = imageHeight - point_left_org[-1]
        point_right_org[-1] = imageHeight - point_right_org[-1]
    return point_eCe_org, point_left_org, point_right_org

def get_json_points_distance(filename, f_cof, point_eCe, point_youzhi_left, point_youzhi_right):
        
    point_eCe_org, point_left_org, point_right_org = get_points_from_jsonFile(filename)
    distance_eCe = np.sqrt((point_eCe_org[0] - point_eCe[0])**2 + (point_eCe_org[1] - point_eCe[1])**2)
    distance_left = np.sqrt((point_left_org[0] - point_youzhi_left[0]) ** 2 + (point_left_org[1] - point_youzhi_left[1]) ** 2)
    distance_right = np.sqrt((point_right_org[0] - point_youzhi_right[0]) ** 2 + (point_right_org[1] - point_youzhi_right[1]) ** 2)
    points = [[point_eCe_org, point_eCe], [point_left_org, point_youzhi_left], [point_right_org, point_youzhi_right]]
    
    return [distance_eCe * f_cof, distance_left * f_cof, distance_right * f_cof], points

def get_thickness_pointFromJsonFile(filename, img_array_pred, img, f_cof, class_numb=5, label_youzhi=2, label_eCe=3, label_target=3):
    
    nHeight = img_array_pred.shape[0]
    
    def plot_pics_thickness(elements_target, k, b):
        x_min = int(elements_target[0].min() - 10)
        x_max = int(elements_target[0].max() + 10)
        x = np.array([i for i in range(x_min, x_max)])
        y = k*x + b
        plt.plot(x, nHeight-y)
        # plt.plot(elements_target[0], elements_target[1], 'o')
        # plt.show()

    coordinate, elements_target, elements_eCe, elements_youzhi, point_eCe, point_youzhi_left, point_youzhi_right = get_coordinate_elements(img_array_pred, class_numb, label_target, label_youzhi, label_eCe)
    point_eCe, point_youzhi_left, point_youzhi_right = get_points_from_jsonFile(filename)
    elements = elements_target
    plt.plot(elements_target[0], elements_target[1], 'bo')
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
    
    # f_cof = 0.1976227  # 1个像素点为0.197mm
    dis_n = 2 # 每2毫米测一次骨璧厚度
    n1, n2 = np.ceil(distance_min * f_cof / dis_n) , np.ceil(distance_max * f_cof / dis_n)
    thickness_all_cej = []
    dis_out = []
    plot_pics_thickness(elements_target, k, b)
    # 计算额侧骨璧厚度，从下（2点线）每隔2mm向上
    for i in range(int(n1), int(n2)):
        h = dis_n/f_cof * i
        b_new = b + h / cos_k
        plot_pics_thickness(elements_target, k, b_new)
        thickness = compute_thickness(elements, k, b_new, f_cof, color='r+')
        thickness_all_cej.append(thickness)
        dis_out.append(h*f_cof)
        # plt.show()
    # plt.show()

    # 计算额侧骨璧厚度，从上点（2点线在1点-即额侧骨璧左侧点）每隔2mm向下
    x_0, y_0 = point_eCe  # 第一个点为根尖点
    # x_0, y_0 = 508.7, 362.5
    b_0 = y_0 - k*x_0
    distance = abs(k * elements[0] + b_0 - elements[1]) / np.sqrt(k ** 2 + 1)
    distance_max = distance.max()
    n2 = np.ceil(distance_max * f_cof / dis_n)
    thickness_all_roof = []
    for i in range(int(n2)):
        h = dis_n / f_cof * i
        b_new = b_0 - h / cos_k
        thickness = compute_thickness(elements, k, b_new, f_cof, color='k*')
        thickness_all_roof.append(thickness)
        plot_pics_thickness(elements_target, k, b_new)
    
    # Display: palatal area #
    # x1, y1 = point_youzhi_left
    # x2, y2 = point_youzhi_right
    # x3, y3 = point_eCe
    
    # io.imshow(img)
    # plt.plot(x1, nHeight-y1, 'r*')
    # plt.plot(x2, nHeight-y2, 'g*')
    # plt.plot(x3, nHeight-y3, 'y*')
    # plt.title('Json')
    # plt.show()
    # End #

    # print(thickness_all_cej, thickness_all_roof)
    return thickness_all_cej, thickness_all_roof, point_eCe, point_youzhi_left, point_youzhi_right


def data_2_excel(filenames, thickness_cej_all, thickness_roof_all, points_distance_all, points_all, saveName):
    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('CEJ厚度', cell_overwrite_ok=True)
    booksheet02 = workbook.add_sheet('根尖——厚度', cell_overwrite_ok=True)
    booksheet03 = workbook.add_sheet('点距离', cell_overwrite_ok=True)
    header = ['姓名', 'CEJ下2-L'	, 'CEJ下4-L', 'CEJ下6-L', 'CEJ下8-L', 'CEJ下10-L']
    header02 = ['姓名', '根尖-腭侧0', '根尖冠向2mm', '根尖-腭侧4mm', '根尖冠向6mm', '根尖冠向8mm']
    header03 = ['姓名', 'apical', 'cej_buccal', 'cej_palatal', 'org-VS-new_apical', 'org-VS-new_left', 'org-VS-new__right']
    for i, item in enumerate(header):
        booksheet.write(0, i, item)
        booksheet02.write(0, i, header02[i])
        booksheet03.write(0, i, header03[i])
    for i, filename in enumerate(filenames):
        booksheet.write(i+1, 0, filename)
        booksheet02.write(i + 1, 0, filename)
        n_youzhi = len(thickness_cej_all[i])
        n_eCe = len(thickness_roof_all[i])
        for j in range(n_youzhi):
            booksheet.write(i + 1, j+1, thickness_cej_all[i][j])
        for j in range(n_eCe):
            booksheet02.write(i + 1, j + 1 , thickness_roof_all[i][j])

        booksheet03.write(i + 1, 0, filename)
        booksheet03.write(i + 1, 1, points_distance_all[i][0])
        booksheet03.write(i + 1, 2, points_distance_all[i][1])
        booksheet03.write(i + 1, 3, points_distance_all[i][2])
        booksheet03.write(i + 1, 4, str(points_all[i][0]))
        booksheet03.write(i + 1, 5, str(points_all[i][1]))
        booksheet03.write(i + 1, 6, str(points_all[i][2]))
    workbook.save(saveName)
    print("\nSaving done!")

def main_get_thickness_4(base_dir, base_dir_json, base_dir_img, class_numb=5, label_youzhi=2, label_eCe=3, label_target=3, isPointJson=True, saveName = 'thickness_result_4.xlsx', save_path_meas=None, save_xlsx=False, savefig=False):

    filenames = os.listdir(base_dir)
    filenames_img = os.listdir(base_dir_img)
    thickness_cej_all, thickness_roof_all = [], []
    points_distance_all, points_all = [], []
    filename_all = []
    
    image_list = glob.glob('./DATA/2labels/test/*image.png')
    label_list = glob.glob('./DATA/2labels/test/*label.png')

    for index, (filename, filename_img, imgpath, labelpath) in enumerate(zip(filenames[:],filenames_img[:],image_list,label_list)):
        # filename = '738_11Maxilla.png'
        print(filename)
        label_img = Image.open(os.path.join(base_dir, filename))
        label_img_array = np.array(label_img)
        
        img = Image.open(os.path.join(base_dir_img, filename_img))
        img_array = np.array(img)
        # a = Image.fromarray(label_img_array*22)
        # a.show()
        unique_array = np.unique(label_img_array)
        print(unique_array)
        # if (1 not in unique_array) or (4 not in unique_array) or (5 not in unique_array):
        #     continue
        f_cof = 0.1976227   # 1个像素点为0.197mm
        jsonFile = os.path.join(base_dir_json, filename.replace('_pred.png', 'label.json'))
        
        if savefig:
            save_path_meas_ = os.path.join(save_path_meas, filename.replace('pred.png', 'meas.png'))
        else:
            save_path_meas_ = None
        
        if isPointJson:
            label_img = Image.open(labelpath)
            label_img_array = np.array(label_img)
            
            img = Image.open(imgpath)
            img_array = np.array(img)
            thickness_cej, thickness_roof, point_eCe, point_youzhi_left, point_youzhi_right = get_thickness_pointFromJsonFile(
                jsonFile, label_img_array, img_array, f_cof, class_numb=class_numb, label_youzhi=label_youzhi, label_eCe=label_eCe, label_target=label_target)
        else:
            thickness_cej, thickness_roof, point_eCe, point_youzhi_left, point_youzhi_right = get_thickness_pointFromDistrict(
                label_img_array, img_array, save_path_meas_, f_cof, class_numb=class_numb, label_youzhi=label_youzhi, label_eCe=label_eCe, label_target=label_target, savefig=savefig)

        distance, points = get_json_points_distance(jsonFile, f_cof, point_eCe, point_youzhi_left, point_youzhi_right)
        points_distance_all.append(distance)
        points_all.append(points)
        thickness_cej_all.append(thickness_cej)
        thickness_roof_all.append(thickness_roof)
        filename_all.append(filename)

        print(index, filename, thickness_cej, thickness_roof)
        pass
    if save_xlsx:
        data_2_excel(filename_all, thickness_cej_all, thickness_roof_all, points_distance_all, points_all, saveName)
    
    return thickness_cej_all, thickness_roof_all


def cal_rmsd_numpy(coord_1, coord_2):
    rmsd = np.sqrt(((np.array(coord_1) - np.array(coord_2)) ** 2).mean())    ## this would be the formula
    return rmsd


def cal_mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

### Single test ###
# filename = '809_Maxilla_1'
# # img_array_pred = np.load('./results/raw_data/result_800_Maxilla_1_image.npy')
# img_array_pred = cv2.imread('./results/raw_data/'+filename+'_pred.png')
# img = cv2.imread('./results/'+filename+'_image.png')
# # img_array_pred = img_array_pred/255
# # print('\nmax img_array_pred: ',np.amax(img_array_pred))

# nHeight = img_array_pred.shape[0]

# class_numb = 3
# label_target = 1
# label_youzhi = 2
# label_eCe = 1

# coordinate, elements_target, elements_eCe, elements_youzhi, point_eCe, point_youzhi_left, point_youzhi_right = get_coordinate_elements(img_array_pred, class_numb, label_target, label_youzhi, label_eCe)

# elements = elements_target
# # plt.plot(elements_target[0], elements_target[1], 'bo')
# x1, y1 = point_youzhi_left
# x2, y2 = point_youzhi_right
# x3, y3 = point_eCe

# io.imshow(img)
# plt.plot(x1, nHeight-y1, 'r*')
# plt.plot(x2, nHeight-y2, 'g*')
# plt.plot(x3, nHeight-y3, 'y*')
# plt.show()

# f_cof = 0.1976227   # 1个像素点为0.197mm

# thickness_cej, thickness_roof, point_eCe, point_youzhi_left, point_youzhi_right = get_thickness_pointFromDistrict(
#     img_array_pred, f_cof, class_numb=class_numb, label_youzhi=label_youzhi, label_eCe=label_eCe, label_target=label_target)

### End ###

### All ###
class_numb = 3
label_target = 1
label_youzhi = 2
label_eCe = 1

save_xlsx = True
method = 'DeepLabV3-resnet50'

savefig = True        # Save thickness measurement plots

base_dir = r'./results/raw_data/'+method+'/preds'
base_dir_json = r'./DATA/9labels/neu_json'
base_dir_img = r'./results/raw_data/'+method+'/imgs'

save_path_preds = './results/raw_data/'+method+'/xlsx/thickness_result_4_preds_Komp.xls'
save_path_json = './results/raw_data/'+method+'/xlsx/thickness_result_4_json.xls'

save_path_meas = './results/raw_data/'+method+'/measure_Komp'

if not os.path.exists(save_path_meas):
    os.mkdir(save_path_meas)

thickness_cej_all, thickness_roof_all = main_get_thickness_4(base_dir, base_dir_json, base_dir_img, class_numb=class_numb, label_youzhi=label_youzhi, label_eCe=label_eCe, label_target=label_target, isPointJson=False, saveName=save_path_preds, save_path_meas=save_path_meas, save_xlsx=save_xlsx, savefig=savefig)
thickness_cej_all_js, thickness_roof_all_js = main_get_thickness_4(base_dir, base_dir_json, base_dir_img, class_numb=class_numb, label_youzhi=label_youzhi, label_eCe=label_eCe, label_target=label_target, isPointJson=True, saveName=save_path_json, save_xlsx=save_xlsx, savefig=False)

""" Compare and evaluations """
# thickness_cej_all_eval, thickness_roof_all_eval = [], []
# thickness_cej_all_js_eval, thickness_roof_all_js_eval = [], []
RMSD_cej, RMSD_roof = [], []
RMSE_cej, RMSE_roof = [], []
MAE_cej, MAE_roof = [], []
thickness_cej_2_eval, thickness_cej_4_eval, thickness_cej_6_eval = [], [], []
thickness_roof_0_eval, thickness_roof_2_eval, thickness_roof_4_eval = [], [], []
thickness_cej_js_2_eval, thickness_cej_js_4_eval, thickness_cej_js_6_eval = [], [], []
thickness_roof_js_0_eval, thickness_roof_js_2_eval, thickness_roof_js_4_eval = [], [], []
cc = 0
for ii in range(len(thickness_cej_all)):
    try:
        if not (np.isnan(float(thickness_cej_all[ii][0])) or np.isnan(float(thickness_cej_all_js[ii][0]))):
            thickness_cej_2_eval.append(thickness_cej_all[ii][0])
            thickness_cej_js_2_eval.append(thickness_cej_all_js[ii][0])
        if not (np.isnan(float(thickness_cej_all[ii][1])) or np.isnan(float(thickness_cej_all_js[ii][1]))):
            thickness_cej_4_eval.append(thickness_cej_all[ii][1])
            thickness_cej_js_4_eval.append(thickness_cej_all_js[ii][1])
        if not (np.isnan(float(thickness_cej_all[ii][2])) or np.isnan(float(thickness_cej_all_js[ii][2]))):
            thickness_cej_6_eval.append(thickness_cej_all[ii][2])
            thickness_cej_js_6_eval.append(thickness_cej_all_js[ii][2])
        if not (np.isnan(float(thickness_roof_all[ii][0])) or np.isnan(float(thickness_roof_all_js[ii][0]))):
            thickness_roof_0_eval.append(thickness_roof_all[ii][0])
            thickness_roof_js_0_eval.append(thickness_roof_all_js[ii][0])
        if not (np.isnan(float(thickness_roof_all[ii][1])) or np.isnan(float(thickness_roof_all_js[ii][1]))):
            thickness_roof_2_eval.append(thickness_roof_all[ii][1])
            thickness_roof_js_2_eval.append(thickness_roof_all_js[ii][1])
        if not (np.isnan(float(thickness_roof_all[ii][2])) or np.isnan(float(thickness_roof_all_js[ii][2]))):
            thickness_roof_4_eval.append(thickness_roof_all[ii][2])
            thickness_roof_js_4_eval.append(thickness_roof_all_js[ii][2])
    except:
        cc += 1
        continue

# rmsd_cej2 = cal_rmsd_numpy(thickness_cej_2_eval, thickness_cej_js_2_eval)
# rmsd_cej4 = cal_rmsd_numpy(thickness_cej_4_eval, thickness_cej_js_4_eval)
# rmsd_cej6 = cal_rmsd_numpy(thickness_cej_6_eval, thickness_cej_js_6_eval)
# rmsd_roof0 = cal_rmsd_numpy(thickness_roof_0_eval, thickness_roof_js_0_eval)
# rmsd_roof2 = cal_rmsd_numpy(thickness_roof_2_eval, thickness_roof_js_2_eval)
# rmsd_roof4 = cal_rmsd_numpy(thickness_roof_4_eval, thickness_roof_js_4_eval)
mae_cej2 = cal_mae(thickness_cej_2_eval, thickness_cej_js_2_eval)
mae_cej4 = cal_mae(thickness_cej_4_eval, thickness_cej_js_4_eval)
mae_cej6 = cal_mae(thickness_cej_6_eval, thickness_cej_js_6_eval)
mae_roof0 = cal_mae(thickness_roof_0_eval, thickness_roof_js_0_eval)
mae_roof2 = cal_mae(thickness_roof_2_eval, thickness_roof_js_2_eval)
mae_roof4 = cal_mae(thickness_roof_4_eval, thickness_roof_js_4_eval)
rmse_cej2 = math.sqrt(compare_mse(np.array(thickness_cej_2_eval), np.array(thickness_cej_js_2_eval)))
rmse_cej4 = math.sqrt(compare_mse(np.array(thickness_cej_4_eval), np.array(thickness_cej_js_4_eval)))
rmse_cej6 = math.sqrt(compare_mse(np.array(thickness_cej_6_eval), np.array(thickness_cej_js_6_eval)))
rmse_roof0 = math.sqrt(compare_mse(np.array(thickness_roof_0_eval), np.array(thickness_roof_js_0_eval)))
rmse_roof2 = math.sqrt(compare_mse(np.array(thickness_roof_2_eval), np.array(thickness_roof_js_2_eval)))
rmse_roof4 = math.sqrt(compare_mse(np.array(thickness_roof_4_eval), np.array(thickness_roof_js_4_eval)))

# print("\nRMSD_cej2: {}, RMSD_cej4: {}, RMSD_cej6: {}".format(rmsd_cej2,rmsd_cej4,rmsd_cej6))
# print("\nRMSD_apical0: {}, RMSD_apical2: {}, RMSD_apical4: {}".format(rmsd_roof0,rmsd_roof2,rmsd_roof4))
print("\nMAE_cej2: {}, MAE_cej4: {}, MAE_cej6: {}".format(mae_cej2,mae_cej4,mae_cej6))
print("\nMAE_apical0: {}, MAE_apical2: {}, MAE_apical4: {}".format(mae_roof0,mae_roof2,mae_roof4))
print("\nRMSE_cej2: {}, RMSE_cej4: {}, RMSE_cej6: {}".format(rmse_cej2,rmse_cej4,rmse_cej6))
print("\nRMSE_apical0: {}, RMSE_apical2: {}, RMSE_apical4: {}".format(rmse_roof0,rmse_roof2,rmse_roof4))

print("\n{} Skipped!".format(cc))
""" End """

### End ###