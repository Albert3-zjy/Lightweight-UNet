'''
    用于数据的k折验证划分
'''
from imblearn.over_sampling import RandomOverSampler, SMOTE
from pandas import Series, DataFrame
from sklearn.model_selection import  StratifiedKFold, KFold, StratifiedGroupKFold
import pandas as pd
import random
import numpy as np
import torch
random.seed(1998)
device = [torch.device('cuda:0'), torch.device('cuda:1')]
label_dict = {
    1 : '湿证判断',
    2 : '湿证程度（整体）',
    3 : '湿证类型',
    4 : '虚实性质'
}

def kFold(args):
    csv_file ='./data/refine_data_20221227.csv'
    frame = pd.read_csv(csv_file, encoding='utf-8')

    X = frame['img_id']
    # label1 = frame['湿证判断'] # y是label，选择你要进行切分的label
    # label2 = frame['湿证程度（整体）']
    # label3 = frame['湿证类型']
    # label4 = frame['虚实性质']
    label = frame[label_dict[args.label_idx]]
    # label_cnt = np.bincount(label)
    # print("class cnt: ", label_cnt)

    # print(X.shape,label1.shape)   #---> (3705,) (3705,)
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    i=0
    global_cnt=[];global_train_weight=[];global_val_weight=[]
    for train_index,test_index in skf.split(X,label):
        print('i=:',i,'train_index:',train_index.shape,'test_index:',test_index.shape)
        X_train,X_test = X[train_index],X[test_index]
        label_train,label_test = label[train_index],label[test_index]

        global_cnt.append(np.bincount(np.asarray(label_train)))
        train_weight = torch.tensor(cal_weights(np.bincount(np.asarray(label_train))), dtype=torch.float32).to(device[0])
        val_weight = torch.tensor(cal_weights(np.bincount(np.asarray(label_test))), dtype=torch.float32).to(device[0])
        global_train_weight.append([train_weight[label] for label in label_train])
        global_val_weight.append([val_weight[label] for label in label_test])
        # ros = RandomOverSampler(random_state=0)
        # X_resampled, y_resampled = ros.fit_resample(DataFrame(X_train.values.reshape(-1, 1)), label_train)
        # print(np.bincount(np.asarray(y_resampled)))
        # label2_train,label2_test = label2[train_index],label2[test_index]
        # label3_train,label3_test = label3[train_index],label3[test_index]
        # label4_train,label4_test = label4[train_index],label4[test_index]
        i += 1
        # train = pd.concat([X_train, label1_train, label2_train, label3_train, label4_train],axis=1)
        # test =pd.concat([X_test, label1_test, label2_test, label3_test, label4_test],axis=1)
        # train = pd.concat([X_resampled, y_resampled], axis=1)
        train = pd.concat([X_train, label_train], axis=1)
        test = pd.concat([X_test, label_test], axis=1)
        train.to_csv('./kFold_csv/label{}_Train_fold_{}.csv'.format(args.label_idx, str(i)))
        test.to_csv('./kFold_csv/label{}_Val_fold_{}.csv'.format(args.label_idx, str(i)))

    return global_cnt, global_train_weight, global_val_weight

def cal_weights(count):
    weights = count / count.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()
    return weights

if __name__ == "__main__":
    csv_file ='./data/refine_data_20221227.csv'
    frame = pd.read_csv(csv_file, encoding='utf-8')

    X = frame['img_id']
    label = frame[label_dict[3]]
    print(np.bincount(np.asarray(label)))

