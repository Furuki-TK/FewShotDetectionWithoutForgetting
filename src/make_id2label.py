from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
from dataloader import PascalVOC, FewShotDataloader

train_split, test_split = 'train', 'val'
dataset_train = PascalVOC(phase=train_split)

f = open('id2label.txt', 'w')

for i in range(20):
    label_name = dataset_train.data.imgs[dataset_train.label2ind[i][0]][0].split('/')[4]
    if '_' in label_name:
        label_name = label_name.split('_')[0]
    f.write(str(i))
    f.write(",")
    f.write(label_name)
    f.write("\n")

f.close()
