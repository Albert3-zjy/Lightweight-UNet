import os
import torch
import torchvision.transforms as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import  StratifiedKFold, KFold, StratifiedGroupKFold

# unloader = tf.ToPILImage()

class generate_data(Dataset):
    # root: 图片目录
    # data_list: 读取csv转化为list后的数据
    # label_index: 要哪个列的label
    def __init__(self, image_path, labels_path, data_list, transforms=None):
        self.image_path = image_path
        self.labels_path = labels_path
        self.transforms = transforms
        self.data_list = data_list

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.data_list[index]))
        label = Image.open(os.path.join(self.labels_path, self.data_list[index]))
        image = np.array(image)
        label = np.array(label, dtype=np.int64)
        # print(np.max(label))
        if self.transforms is not None:
            image = self.transforms(image)
            label = self.transforms(label)
        #     # img_copy = img.clone()
        #     # img_copy = unloader(img)
        #
        #     # img.show()
        #     # pyplot.imshow(img_copy)

        # return image, label
        return {
            'image': image,
            'mask': label
        }

    def __len__(self):
        return len(self.data_list)

def kFold(image_path, labels_path):
    train_list = [];  test_list = [];
    image_files = np.array(os.listdir(image_path))
    labels_files = np.array(os.listdir(labels_path))
    num_files = len(image_files)
    assert len(image_files) == len(labels_files)
    # print("num_files: ", num_files)
    # print(image_files, labels_files)

    skf = KFold(n_splits=5, shuffle=True)
    i = 0
    for train_index, test_index in skf.split(image_files, labels_files):
        print('i=:', i, 'train_index:', train_index.shape, 'test_index:', test_index.shape)
        image_train, image_test = image_files[train_index], image_files[test_index]
        labels_train, labels_test = labels_files[train_index], labels_files[test_index]

        # print(len(image_train), len(image_test))
        # print(len(labels_train), len(labels_test))
        train_list.append(image_train)
        test_list.append(image_test)
        train_csv = pd.DataFrame({"img": train_list[i]})
        test_csv = pd.DataFrame({"img": test_list[i]})
        train_csv.to_csv('/data33/23/jiayang/python_code/example_project/RegionSegmentation/Pytorch-UNet/csv_files/Train_fold_{}.csv'.format(str(i)), index=None)
        test_csv.to_csv('/data33/23/jiayang/python_code/example_project/RegionSegmentation/Pytorch-UNet/csv_files/Val_fold_{}.csv'.format(str(i)), index=None)
        # print(train_list, test_list)
        i += 1

    return train_list, test_list

if __name__ == "__main__":
    image_path = '/data33/23/jiayang/python_code/example_project/RegionSegmentation/Tooth_dataset/training/input'
    labels_path = '/data33/23/jiayang/python_code/example_project/RegionSegmentation/Tooth_dataset/training/output'
    train_list, test_list = kFold(image_path, labels_path)

    train_transforms = tf.Compose([
        tf.ToTensor()
    ])
    test_transforms = tf.Compose([
        tf.ToTensor()
    ])

    train_dataset = generate_data(image_path, labels_path, train_list[0], train_transforms)
    test_dataset = generate_data(image_path, labels_path, test_list[0], test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    train_batch = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_batch['image'].size()}")
    print(f"Labels batch shape: {train_batch['mask'].size()}")
    print(f"Labels batch info: {torch.max(train_batch['mask'])}")