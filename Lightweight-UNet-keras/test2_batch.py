import os
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from PIL import Image
from skimage import io
from evaluations import *
from scipy.spatial.distance import directed_hausdorff
import cv2
import glob
from U2NET import BatchNorm
import time


def read_png_and_preprocess(path, channels, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=channels)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32)/255.0
    return img


def getcount(y_array, x_array):
    myset = list(set(y_array))
    myset.sort()
    list_y = [list(y_array).count(item) for item in myset]
    return myset, list_y


# weights_num = 'UNET_crop'  #保持一直，但是不要有.h5后缀
# weights_num = 'U2NET_crop'
weights_num = 'LUNet_crop'
# weights_num = 'LUNet_crop_for_test_try'
# weights_num = 'DeepLabV3p-resnet50'
# weights_num = 'DeepLabV3p-resnet101'
# weights_num = 'DeepLabV3p-mobV2'
# weights_num = 'DeepLabV3p-Xception'

# model = tf.keras.models.load_model("./weights/2labels/"+weights_num+".h5")   #旧语句
model = tf.keras.models.load_model("./weights/2labels/"+weights_num+".h5")

num_classes = 2

# mode = 'valid'          # 测试集：'test'，验证集：'valid'
mode = 'valid'
if mode == 'test': 
    save_path = "./test_imgs/2labels/"+weights_num
elif mode == 'valid':
    save_path = "./test_imgs/2labels/"+weights_num+'_val'
else:
    raise ValueError

save_raw_path = save_path+'/raw_data'
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(save_raw_path):
    os.mkdir(save_raw_path)

# model.summary()
# while(1):
    
# imgnum = input("图像num:")
# imgnum = 5
DICE1, DICE2, DICE3, DICE4 = [], [], [], []
MIOU1, MIOU2, MIOU3, MIOU4 = [], [], [], []
HAUS1, HAUS2, HAUS3, HAUS4 = [], [], [], []

if mode == 'test':
    image_list = glob.glob('../DATA/2labels/test_crop/*image.png')
    label_list = glob.glob('../DATA/2labels/test_crop/*label.png')
elif mode == 'valid':
    image_list = glob.glob('../DATA/2labels/valid_crop/*image.png')
    label_list = glob.glob('../DATA/2labels/valid_crop/*label.png')
else:
    raise ValueError

time_cnt = 0
print("\nlength: ",len(image_list))
for imgnum in range(len(image_list)):
    # imgpath = "./data_palatal/test/" + str(imgnum) + "_image.png"
    imgpath = image_list[imgnum]
    fpath, fname = os.path.split(imgpath)
    # 读取Label #
    # labelpath = "./data_palatal/test/" + str(imgnum) + "_label.png"
    labelpath = label_list[imgnum]

    label = cv2.imread(labelpath)
    # print(labelpath)

    if len(label.shape) == 3:
        label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
    # End #
    # H, W = 587, 945
    # H, W = 507, 860
    H, W = 320, 384

    # For old trained UNET #
    # image_size = (304, 480)
    # input_image = read_png_and_preprocess(imgpath, 3, image_size)
    # For new model eg.LUNet #
    image_size = (320, 384)
    input_image = read_png_and_preprocess(imgpath, 3, image_size)
    # End #

    time_st = time.time()
    pred_mask = model.predict(input_image[tf.newaxis, ...]) # -1*256*256*2
    # pred_mask = input_image[tf.newaxis, ...]
    print("\n1")
    ### 计算耗时 ###
    time_end = time.time()
    time_consump = time_end - time_st
    time_cnt += time_consump
    ### End ###

    pred_mask = tf.argmax(pred_mask, axis=-1) # -1*256*256
    pred_mask = tf.squeeze(pred_mask, 0) # 256*256
    pred_mask = pred_mask[..., tf.newaxis] # 256*256*1

    pred_mask = tf.image.resize(pred_mask, size=(H, W), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # points_2 = np.array(tf.where(pred_mask == 2))
    # points_6 = np.array(tf.where(pred_mask == 6))
    # points_2 = np.array(tf.where(pred_mask == 1))
    points_6 = np.array(tf.where(pred_mask == 1))
    points_3 = np.array(tf.where(pred_mask == 2))
    # points_4 = np.array(tf.where(pred_mask == 4))

    ### 评价指标 ###
    pred_1, pred_2, pred_3, pred_4 = np.zeros((H,W)), np.zeros((H,W)), np.zeros((H,W)), np.zeros((H,W))
    label_1, label_2, label_3, label_4 = np.zeros((H,W)), np.zeros((H,W)), np.zeros((H,W)), np.zeros((H,W))
    pred_hyb = np.zeros((H,W))
    # pred_1[points_2[:,0], points_2[:,1]] = 1
    pred_2[points_6[:,0], points_6[:,1]] = 1
    pred_3[points_3[:,0], points_3[:,1]] = 1
    # pred_4[points_4[:,0], points_4[:,1]] = 1

    pred_hyb[points_6[:,0], points_6[:,1]] = 1
    pred_hyb[points_3[:,0], points_3[:,1]] = 2

    # label_1[label==1] = 1
    label_2[label==1] = 1
    label_3[label==2] = 1
    # label_4[label==4] = 1

    # plt.subplot(1,2,1)
    # io.imshow(pred_2*255)
    # plt.subplot(1,2,2)
    # io.imshow(label*255)
    # plt.show()

    # dice1 = single_dice_coef(pred_1, label_1)
    dice2 = dice(pred_2, label_2, n_class=num_classes)
    dice3 = dice(pred_3, label_3, n_class=num_classes)
    # dice4 = single_dice_coef(pred_4, label_4)
    # miou1 = iou(label_1, pred_1, n_class=num_classes)
    miou2 = iou(label_2, pred_2, n_class=num_classes)
    miou3 = iou(label_3, pred_3, n_class=num_classes)
    # miou4 = iou(label_4, pred_4, n_class=num_classes)
    # print("\nDice 1: {}, Dice 2: {}".format(dice1, dice2))
    # print("\nmIoU 1: {}, mIoU 2: {}".format(miou1, miou2))
    # haus1 = directed_hausdorff(label_1, pred_1)
    haus2 = directed_hausdorff(label_2, pred_2)
    haus3 = directed_hausdorff(label_3, pred_3)
    # haus4 = directed_hausdorff(label_4, pred_4)

    # DICE1.append(dice1)
    DICE2.append(dice2)
    DICE3.append(dice3)
    # DICE4.append(dice4)
    # MIOU1.append(miou1)
    MIOU2.append(miou2)
    MIOU3.append(miou3)
    # MIOU4.append(miou4)
    # HAUS1.append(haus1)
    HAUS2.append(haus2)
    HAUS3.append(haus3)
    # HAUS4.append(haus4)

    ### End ###

    # line_2_x, line_2_y = getcount(points_2[:,0], points_2[:,1])
    line_6_x, line_6_y = getcount(points_6[:,0], points_6[:,1])
    line_3_x, line_3_y = getcount(points_3[:,0], points_3[:,1])
    # line_4_x, line_4_y = getcount(points_4[:,0], points_4[:,1])

    # plt.plot(line_2_x, line_2_y, 'b', label='buccal')
    # plt.plot(line_6_x, line_6_y, 'g', label='palatal')
    # plt.legend(loc="upper right")
    # plt.xlabel("Y")
    # plt.ylabel("COUNT")

    #    plt.savefig("./count.png")
    #    plt.close('all')

    # 把预测的点放到图上显示
    result_img_array = np.array(Image.open(imgpath))
    # H_test, W_test, _ = result_img_array.shape
    # if not(H_test == H and W_test == H):
        # result_img_array = resize(result_img_array, (H, W), mode='constant', preserve_range=True)

    # for pointEin in points_2:
        # for i in range(-1,1):
            # result_img_array[pointEin[0], pointEin[1]+i, 0] = 0
            # result_img_array[pointEin[0], pointEin[1]+i, 1] = 255
            # result_img_array[pointEin[0], pointEin[1]+i, 2] = 0
        # del pointEin
    for pointZwei in points_6:
        for i in range(-1,1):
            result_img_array[pointZwei[0], pointZwei[1]+i, 0] = 255
            result_img_array[pointZwei[0], pointZwei[1]+i, 1] = 0
            result_img_array[pointZwei[0], pointZwei[1]+i, 2] = 0
        del pointZwei
    for pointTrei in points_3:
        for i in range(-1,1):
            result_img_array[pointTrei[0], pointTrei[1]+i, 0] = 0
            result_img_array[pointTrei[0], pointTrei[1]+i, 1] = 255
            result_img_array[pointTrei[0], pointTrei[1]+i, 2] = 255
        del pointTrei
    # for pointVier in points_4:
        # for i in range(-1,1):
            # result_img_array[pointVier[0], pointVier[1]+i, 0] = 255
            # result_img_array[pointVier[0], pointVier[1]+i, 1] = 0
            # result_img_array[pointVier[0], pointVier[1]+i, 2] = 255
        # del pointVier
    ### 保存结果 ###
    # 可视化 #
    # result_img = Image.fromarray(result_img_array)
    # result_img.save("./test_imgs/2labels/"+weights_num+"/result_"+fname[:-4]+".png")
    result_img = result_img_array
    cv2.imwrite(save_path+"/result_"+fname[:-4]+".png",result_img)
    # count_img = Image.open("./count.png")
    # 原始结果数据 #
    fname1 = fname.replace('image', 'pred')
    # np.save(save_raw_path+"/"+fname1[:-4]+".npy",result_img, pred_mask)
    # tf.keras.preprocessing.image.save_img(save_raw_path+"/"+fname1[:-4]+".png", pred_mask)
    cv2.imwrite(save_raw_path+"/"+fname1[:-4]+".png", pred_hyb)
    # End #
    # result_img.show()
    #count_img.show()
    ### End ###
time_consump_avg = time_cnt / len(image_list)
dice1_mean, dice2_mean = 0, np.mean(DICE2)
miou1_mean, miou2_mean = 0, np.mean(MIOU2)
haus1_mean, haus2_mean = 0, np.mean(HAUS2)
dice3_mean, dice4_mean = np.mean(DICE3), 0
miou3_mean, miou4_mean = np.mean(MIOU3), 0
haus3_mean, haus4_mean = np.mean(HAUS3), 0

print("\nDice 1 mean: {}, Dice 2 mean: {}".format(dice1_mean, dice2_mean))
print("\nmIoU 1 mean: {}, mIoU 2 mean: {}".format(miou1_mean, miou2_mean))
print("\nhaus 1 mean: {}, haus 2 mean: {}".format(haus1_mean, haus2_mean))
print("\nDice 3 mean: {}, Dice 4 mean: {}".format(dice3_mean, dice4_mean))
print("\nmIoU 3 mean: {}, mIoU 4 mean: {}".format(miou3_mean, miou4_mean))
print("\nhaus 3 mean: {}, haus 4 mean: {}".format(haus3_mean, haus4_mean))
print("\nAvg time consumption for 1 img: ", time_consump_avg, "sec")
