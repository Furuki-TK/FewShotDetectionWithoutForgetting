# -*- coding: utf-8 -*-
from __future__ import print_function

import shutil
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


dataset_train = torchvision.datasets.VOCDetection("../../datasets/PASCAL3D+_release1.1/PASCAL")
dataset_test_path = './datasets/VOCDetection/test'


num_data_train = len(dataset_train)#VOCデータ数

f = open('test_VOCimage_name.txt', 'w')

for i in range(num_data_train):
    if 'name' in dataset_train[i][1]['annotation']['object']:
        if dataset_train[i][1]['annotation']['object']['name'] == 'pottedplant' or dataset_train[i][1]['annotation']['object']['name'] == 'train' or dataset_train[i][1]['annotation']['object']['name'] == 'sheep' or dataset_train[i][1]['annotation']['object']['name'] == 'sofa' or dataset_train[i][1]['annotation']['object']['name'] == 'tvmoniter':
            print(dataset_train[i][1]['annotation']['filename'],' is hit')
            f.write(dataset_train[i][1]['annotation']['filename'])
            f.write("\n")
            # new_path = os.path.join(dataset_test_path,dataset_train[i][1]['annotation']['filename'])
            # shutil.copyfile(dataset_train[i][1]['annotation']['filename'], new_path)

    elif 'name' not in dataset_train[i][1]['annotation']['object']:
        for j in range(len(dataset_train[i][1]['annotation']['object'])):
            if dataset_train[i][1]['annotation']['object'][j]['name'] == 'pottedplant' or dataset_train[i][1]['annotation']['object'][j]['name'] == 'train' or dataset_train[i][1]['annotation']['object'][j]['name'] == 'sheep' or dataset_train[i][1]['annotation']['object'][j]['name'] == 'sofa' or dataset_train[i][1]['annotation']['object'][j]['name'] == 'tvmoniter':
                print(dataset_train[i][1]['annotation']['filename'],' is hit')
                f.write(dataset_train[i][1]['annotation']['filename'])
                f.write("\n")
                break;

f.close()
