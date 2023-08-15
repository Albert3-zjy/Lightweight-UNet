import os
from shutil import copyfile
import numpy as np
import pandas as pd
from matplotlib import pyplot
from PIL import Image

if __name__ == "__main__":
    csv_folder = "/data33/23/jiayang/python_code/example_project/RegionSegmentation/Pytorch-UNet/csv_files"
    dst_folder = "/data33/23/jiayang/python_code/example_project/RegionSegmentation/Tooth_dataset_folds"
    image_file_folder = "/data33/23/jiayang/python_code/example_project/RegionSegmentation/Tooth_dataset/training/input"
    label_file_folder = "/data33/23/jiayang/python_code/example_project/RegionSegmentation/Tooth_dataset/training/output"

    for fold in range(0, 5):
        train_csv_file = f"{csv_folder}/Train_fold_{fold}.csv"
        val_csv_file = f"{csv_folder}/Val_fold_{fold}.csv"
        
        train_frame = pd.read_csv(train_csv_file, encoding='utf-8')
        val_frame = pd.read_csv(val_csv_file, encoding='utf-8')

        train_X = np.asarray(train_frame['img'])
        val_X = np.asarray(val_frame['img'])

        if not os.path.exists(f"{dst_folder}/fold_{fold}"):
            os.mkdir(f"{dst_folder}/fold_{fold}")
            os.mkdir(f"{dst_folder}/fold_{fold}/train")
            os.mkdir(f"{dst_folder}/fold_{fold}/test")
            os.mkdir(f"{dst_folder}/fold_{fold}/train/image")
            os.mkdir(f"{dst_folder}/fold_{fold}/train/anno")
            os.mkdir(f"{dst_folder}/fold_{fold}/test/image")
            os.mkdir(f"{dst_folder}/fold_{fold}/test/anno")

        dst_train = f"{dst_folder}/fold_{fold}/train"
        dst_val = f"{dst_folder}/fold_{fold}/test"

        for relative_file_name in train_X:
            image_filename = f"{image_file_folder}/{relative_file_name}"
            label_filename = f"{label_file_folder}/{relative_file_name}"

            image_dst_filename = f"{dst_train}/image/{relative_file_name}"
            label_dst_filename = f"{dst_train}/anno/{relative_file_name}"

            copyfile(image_filename, image_dst_filename)
            copyfile(label_filename, label_dst_filename)

        for relative_file_name in val_X:
            image_filename = f"{image_file_folder}/{relative_file_name}"
            label_filename = f"{label_file_folder}/{relative_file_name}"

            image_dst_filename = f"{dst_val}/image/{relative_file_name}"
            label_dst_filename = f"{dst_val}/anno/{relative_file_name}"

            copyfile(image_filename, image_dst_filename)
            copyfile(label_filename, label_dst_filename)