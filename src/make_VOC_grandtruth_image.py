# -*- coding: utf-8 -*-
from __future__ import print_function

import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


dataset_train = torchvision.datasets.VOCDetection("../../../datasets/PASCAL3D+_release1.1/PASCAL",image_set='val')
dataset_test_path = '../datasets/VOCDetection'
file_list = []
file_list1 = []
file_list2 = []
count = 0
num_data_train = len(dataset_train)#VOCデータ数

f = open(dataset_test_path+'/valset_c1-20.txt', 'r')

for s in f:
    file_list.append(s.replace('\n',''))

f.close()

g = open(dataset_test_path+'/valset_c1-15.txt', 'r')

for s in g:
    file_list1.append(s.replace('\n',''))

g.close()

for s in file_list:
    if s not in file_list1:
        file_list2.append(s)

print(len(file_list2))

for i in range(num_data_train):
    if dataset_train[i][1]['annotation']['filename'] in file_list2:
        input = dataset_test_path + "/all_images/JPEGImages/" + dataset_train[i][1]['annotation']['filename']
        output = dataset_test_path + "/grandtruth_images/" + dataset_train[i][1]['annotation']['filename']
        img_name = dataset_train[i][1]['annotation']['filename'].split(".")[0]
        grandtruth_txt = dataset_test_path + "/grandtruth_txt/" + img_name + ".txt"

        ff = open(grandtruth_txt, 'w')

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        ax.imshow(np.array( Image.open(input)))

        if 'name' in dataset_train[i][1]['annotation']['object']:
            x_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmin'])
            y_min = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymin'])
            x_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['xmax'])
            y_max = int(dataset_train[i][1]['annotation']['object']['bndbox']['ymax'])

            rect = mpatches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, alpha=0.2, facecolor='red', edgecolor='red', linewidth=1)
            plt.text(x_min, y_min+1 ,"label:"+dataset_train[i][1]['annotation']['object']['name'], fontsize=15)
            ax.add_patch(rect)

            write_txt = dataset_train[i][1]['annotation']['object']['name'] + " " + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) +"\n"
            ff.write(write_txt)

        elif 'name' not in dataset_train[i][1]['annotation']['object']:
            for j in range(len(dataset_train[i][1]['annotation']['object'])):
                x_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmin'])
                y_min = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymin'])
                x_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['xmax'])
                y_max = int(dataset_train[i][1]['annotation']['object'][j]['bndbox']['ymax'])

                rect = mpatches.Rectangle((x_min, y_min),  x_max - x_min, y_max - y_min, alpha=0.2, facecolor='red', edgecolor='red', linewidth=1)
                plt.text(x_min, y_min+1 ,"label:"+dataset_train[i][1]['annotation']['object'][j]['name'], fontsize=15)
                ax.add_patch(rect)

                write_txt = dataset_train[i][1]['annotation']['object'][j]['name'] + " " + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) +"\n"
                ff.write(write_txt)

        plt.savefig(output)
        plt.close()
        ff.close()
        count += 1
        sys.stdout.write("\r%d" % count)
        sys.stdout.flush()
