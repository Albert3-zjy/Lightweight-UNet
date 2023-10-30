import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image, ImageDraw
import cv2


def polygons_to_mask_array(polygons, width = 300, height = 300):
    '''
    This function takes a list of lists that contains polygon masks for each building. Example;
    
    [[x11,y11,x12,y12,...],...,[xn1,yn1,xn2,yn2,...]]
    
    The return of this function is an array of size width x height which contains a binary mask
    as defined by the list of polygons. This will be the target for our network!
    '''
    
    img = Image.new('L', (width, height), 0)    
    ck = 1
    for polygon in polygons:
        ##########################
        # cc = 1
        # for k in polygon['label']:
        #     if cc == 1:
        #         nested_lst_of_tuples = [tuple(l) for l in polygon['points']]
        #         # ImageDraw.Draw(img).polygon(nested_lst_of_tuples, outline=1, fill=1)
        #         ImageDraw.Draw(img).polygon(nested_lst_of_tuples, outline=ck, fill=ck)
        #         ck += 1
        #         # print("ck: ",ck)
        #     cc += 1
        ##########################
        if polygon['label'] == 'buccal':
            cw = 1
        elif polygon['label'] == 'palatal':
            cw = 2
        elif polygon['label'] == 'ena':
            cw = 3
        elif polygon['label'] == 'tooth':
            cw = 4
        else:
            raise (ValueError)
        nested_lst_of_tuples = [tuple(l) for l in polygon['points']]
        # ImageDraw.Draw(img).polygon(nested_lst_of_tuples, outline=1, fill=1)
        ImageDraw.Draw(img).polygon(nested_lst_of_tuples, outline=cw, fill=cw)
        ##########################
    mask = np.array(img)
    
    return mask


json_names = glob.glob('./DATA/9labels/800-814/*.json')
images = glob.glob('./DATA/4labels/800-814/*.jpeg')

save_dir = './DATA/4labels/imgs4/'
save_neu_json_dir = './DATA/9labels/neu_json/'

# H, W = 945, 587
H, W = 860, 507

ext = '.png'

_, aa0 = os.path.split(json_names[0])
num0 = int(aa0[:3])
num_prec = 0
cc = 1
for ii in range(len(json_names)):
    with open(json_names[ii], encoding = 'utf-8') as f:
        data = json.load(f)
        
    _, aa = os.path.split(json_names[ii])
    num = int(aa[:3])
    
    numii = num
    if numii < 10:
        iik = '00'+str(numii)
    elif numii < 100:
        iik = '0'+str(numii)
    else:
        iik = str(numii)
    
    # 文件名 #11变成1,12变成2,21变成3,22变成4
    if num == num_prec:
        cc += 1
        img_name = iik+'_Maxilla_'+str(cc)+'_image'+ext
        label_name = iik+'_Maxilla_'+str(cc)+'label'+ext
        json_name = iik+'_Maxilla_'+str(cc)+'label.json'
    else:
        cc = 1
        img_name = iik+'_Maxilla_'+str(cc)+'_image'+ext
        label_name = iik+'_Maxilla_'+str(cc)+'label'+ext
        json_name = iik+'_Maxilla_'+str(cc)+'label.json'
    # End #
    
    save_path_img = os.path.join(save_dir,img_name)
    save_path_lab = os.path.join(save_dir,label_name)
    save_path_neu_json = os.path.join(save_neu_json_dir,json_name)

    # 保存新.json文件 #
    with open(save_path_neu_json,'w',encoding='utf-8') as f:
        json.dump(data, f,ensure_ascii=False)
    # End #
    
    # 保存图像 y 人工标注数据 #
    # img = cv2.imread(images[ii])
    # img = io.imread(images[ii])
    # label = polygons_to_mask_array(data['shapes'], H, W)
    
    # cv2.imwrite(save_path_img, img)
    # cv2.imwrite(save_path_lab, label)
    # End #

    num_prec = num
    # io.imshow(img)
    # plt.show()

print("Saving done!")
#plt.imshow(polygons_to_mask_array(data['shapes'], H, W))

