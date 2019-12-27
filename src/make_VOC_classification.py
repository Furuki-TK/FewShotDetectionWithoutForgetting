# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import os.path
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets

phase = 'val'
output_dir = '/home/datasets/VOCclassification'
class_names = ['aeroplane', 'bicycle','bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog' ,'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

if not os.path.exists(os.path.join(output_dir, phase)):
    os.mkdir(os.path.join(output_dir, phase))

for class_name in class_names:
    if not os.path.exists(os.path.join(output_dir, phase ,class_name)):
        os.mkdir(os.path.join(output_dir, phase ,class_name))

dataset = torchvision.datasets.VOCDetection("/home/datasets/VOCdetection/imageset_trainval",image_set=phase)

num_data = len(dataset)#VOCデータ数
data_count = 0

for i in range(num_data):
    if 'name' in dataset[i][1]['annotation']['object'] and dataset[i][1]['annotation']['object']['occluded'] == '0' and dataset[i][1]['annotation']['object']['difficult'] == '0' and dataset[i][1]['annotation']['object']['truncated'] == '0':
        image_data = cv2.imread(dataset.images[i])
        x_min = int(dataset[i][1]['annotation']['object']['bndbox']['xmin'])
        y_min = int(dataset[i][1]['annotation']['object']['bndbox']['ymin'])
        x_max = int(dataset[i][1]['annotation']['object']['bndbox']['xmax'])
        y_max = int(dataset[i][1]['annotation']['object']['bndbox']['ymax'])
        cv2.imwrite(os.path.join(output_dir, phase ,dataset[i][1]['annotation']['object']['name'] , dataset[i][1]['annotation']['filename']), image_data[y_min:y_max, x_min:x_max])
        data_count += 1
        sys.stdout.write("finish : \r%d" % data_count)
        sys.stdout.flush()

    elif 'name' not in dataset[i][1]['annotation']['object']:
        for j in range(len(dataset[i][1]['annotation']['object'])):
            if dataset[i][1]['annotation']['object'][j]['occluded'] == '0' and dataset[i][1]['annotation']['object'][j]['difficult'] == '0' and dataset[i][1]['annotation']['object'][j]['truncated'] == '0':
                image_data = cv2.imread(dataset.images[i])
                x_min = int(dataset[i][1]['annotation']['object'][j]['bndbox']['xmin'])
                y_min = int(dataset[i][1]['annotation']['object'][j]['bndbox']['ymin'])
                x_max = int(dataset[i][1]['annotation']['object'][j]['bndbox']['xmax'])
                y_max = int(dataset[i][1]['annotation']['object'][j]['bndbox']['ymax'])
                cv2.imwrite(os.path.join(output_dir, phase , dataset[i][1]['annotation']['object'][j]['name'], str(j) + "_" + dataset[i][1]['annotation']['filename']), image_data[y_min:y_max, x_min:x_max])
                data_count += 1
                sys.stdout.write("finish : \r%d" % data_count)
                sys.stdout.flush()
