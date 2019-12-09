# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets

from PIL import Image
from PIL import ImageEnhance

phase = 'train'
output_dir = '../datasets/VOCclassificasion'
class_names = ['aeroplane', 'bicycle','bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog' ,'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

if not os.path.exists(os.path.join(output_dir, phase)):
    os.mkdir(os.path.join(output_dir, phase))

for class_name in class_names:
    if not os.path.exists(os.path.join(output_dir, phase ,class_name)):
        os.mkdir(os.path.join(output_dir, phase ,class_name))

dataset_train = torchvision.datasets.VOCDetection("../../../datasets/PASCAL3D+_release1.1/PASCAL",image_set=phase)

num_data_train = len(dataset_train)#VOCデータ数

for i in range(num_data_train):
    if 'name' in dataset_train[i][1]['annotation']['object'] and dataset_train[i][1]['annotation']['object']['occluded'] == '0' and dataset_train[i][1]['annotation']['object']['difficult'] == '0' and dataset_train[i][1]['annotation']['object']['truncated'] == '0':
        if dataset_train[i][1]['annotation']['object']['name'] == 'sheep':
            print(dataset_train[i][1]['annotation']['filename'],' is sheep')
            image_data = cv2.imread(dataset_train.images[i])
            x_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmin'])
            y_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymin'])
            x_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmax'])
            y_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymax'])
            cv2.imwrite("../../datasets/PASCAL3D+_release1.1/Images/sheep_pascal_trim/" + dataset_train[i][1]['annotation']['filename'], image_data[y_min:y_max, x_min:x_max])
            print("----------------------------------------------")
        elif dataset_train[i][1]['annotation']['object']['name'] == 'horse':
            print(dataset_train[i][1]['annotation']['filename'],' is horse')
            image_data = cv2.imread(dataset_train.images[i])
            x_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmin'])
            y_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymin'])
            x_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmax'])
            y_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymax'])
            cv2.imwrite("../../datasets/PASCAL3D+_release1.1/Images/horse_pascal_trim/" + dataset_train[i][1]['annotation']['filename'], image_data[y_min:y_max, x_min:x_max])
            print("----------------------------------------------")
        elif dataset_train[i][1]['annotation']['object']['name'] == 'cow':
            print(dataset_train[i][1]['annotation']['filename'],' is cow')
            image_data = cv2.imread(dataset_train.images[i])
            x_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmin'])
            y_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymin'])
            x_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmax'])
            y_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymax'])
            cv2.imwrite("../../datasets/PASCAL3D+_release1.1/Images/cow_pascal_trim/" + dataset_train[i][1]['annotation']['filename'], image_data[y_min:y_max, x_min:x_max])
            print("----------------------------------------------")
        elif dataset_train[i][1]['annotation']['object']['name'] == 'dog':
            print(dataset_train[i][1]['annotation']['filename'],' is dog')
            image_data = cv2.imread(dataset_train.images[i])
            x_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmin'])
            y_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymin'])
            x_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmax'])
            y_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymax'])
            cv2.imwrite("../../datasets/PASCAL3D+_release1.1/Images/dog_pascal_trim/" + dataset_train[i][1]['annotation']['filename'], image_data[y_min:y_max, x_min:x_max])
            print("----------------------------------------------")
    elif 'name' not in dataset_train[i][1]['annotation']['object']:
        for j in range(len(dataset_train[i][1]['annotation']['object'])):
            if dataset_train[i][1]['annotation']['object'][j]['occluded'] == '0' and dataset_train[i][1]['annotation']['object'][j]['difficult'] == '0' and dataset_train[i][1]['annotation']['object'][j]['truncated'] == '0':
                if dataset_train[i][1]['annotation']['object'][j]['name'] == 'sheep':
                    print(dataset_train[i][1]['annotation']['filename'],' is sheep')
                    image_data = cv2.imread(dataset_train.images[i])
                    x_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmin'])
                    y_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymin'])
                    x_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmax'])
                    y_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymax'])
                    cv2.imwrite("../../datasets/PASCAL3D+_release1.1/Images/sheep_pascal_trim/" + str(j) + "_" + dataset_train[i][1]['annotation']['filename'], image_data[y_min:y_max, x_min:x_max])
                    print("----------------------------------------------")
                elif dataset_train[i][1]['annotation']['object'][j]['name'] == 'horse':
                    print(dataset_train[i][1]['annotation']['filename'],' is horse')
                    image_data = cv2.imread(dataset_train.images[i])
                    x_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmin'])
                    y_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymin'])
                    x_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmax'])
                    y_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymax'])
                    cv2.imwrite("../../datasets/PASCAL3D+_release1.1/Images/horse_pascal_trim/" + str(j) + "_" + dataset_train[i][1]['annotation']['filename'], image_data[y_min:y_max, x_min:x_max])
                    print("----------------------------------------------")
                elif dataset_train[i][1]['annotation']['object'][j]['name'] == 'cow':
                    print(dataset_train[i][1]['annotation']['filename'],' is cow')
                    image_data = cv2.imread(dataset_train.images[i])
                    x_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmin'])
                    y_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymin'])
                    x_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmax'])
                    y_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymax'])
                    cv2.imwrite("../../datasets/PASCAL3D+_release1.1/Images/cow_pascal_trim/" + str(j) + "_" + dataset_train[i][1]['annotation']['filename'], image_data[y_min:y_max, x_min:x_max])
                    print("----------------------------------------------")
                elif dataset_train[i][1]['annotation']['object'][j]['name'] == 'dog':
                    print(dataset_train[i][1]['annotation']['filename'],' is dog')
                    image_data = cv2.imread(dataset_train.images[i])
                    x_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmin'])
                    y_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymin'])
                    x_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmax'])
                    y_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymax'])
                    cv2.imwrite("../../datasets/PASCAL3D+_release1.1/Images/dog_pascal_trim/" + str(j) + "_" + dataset_train[i][1]['annotation']['filename'], image_data[y_min:y_max, x_min:x_max])
                    print("----------------------------------------------")
