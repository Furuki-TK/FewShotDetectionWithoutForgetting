# -*- coding: utf-8 -*-
import argparse
import os
import imp

import algorithms as alg

import shutil
import urllib
import pickle
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataloader import QuaryData, VOCTrainSet

dataset_path = './datasets/VOCdetection'
voc_path ='VOCdevkit/VOC2012/JPEGImages'

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest = 'image', help =
                            "text file with list of input images name",
                            default = " ", type = str)
    parser.add_argument('--config', type=str, default='pascalVOC_Conv128CosineClassifierGenWeightAvgN5',
        help='config file with parameters of the experiment.')
    parser.add_argument('-id','--testset_id',dest = 'id' ,type=int, default=0,
        help='id is the number from 0 to 7')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')

    return parser.parse_args()

if __name__ == '__main__':
    args_opt = arg_parse()

    exp_config_file = os.path.join('.', 'config', args_opt.config + '.py')
    exp_directory = os.path.join('.', 'experiments', 'Grad-CAM', args_opt.config)
    exp_model_directory = os.path.join('.', 'experiments', args_opt.config)

    # Load the configuration params of the experiment
    print('Launching experiment: %s' % exp_config_file)
    config = imp.load_source("",exp_config_file).config
    config['thres'] = 30.0
    NN = '_' + str(config['thres']) + '%_test'
    config['img_dir'] = '/home/output/GradCam/images'+ NN
    config['txt_dir'] = '/home/output/GradCam/txt' + NN
    config['exp_dir'] = exp_directory
    config['model_dir'] = exp_model_directory
    if not os.path.exists(config['img_dir']):
        os.makedirs(config['img_dir'])

    if not os.path.exists(config['txt_dir']):
        os.makedirs(config['txt_dir'])

    print('Loading experiment %s from file: %s' %
          (args_opt.config, exp_config_file))
    print('Generated logs, snapshots, and model files will be stored on %s' %
          (config['exp_dir']))


    print("Loading network.....")
    algorithm = alg.FewShotClassification(config)


    print("Loading checkpoint.....")
    algorithm.load_checkpoint(epoch='*', train=False, suffix='.best')

    if args_opt.cuda: # enable cuda
        algorithm.load_to_gpu()

    print("Entry Support Data.....")
    if 'VOC' in args_opt.config:
        data_support = VOCTrainSet(phase='train', category=5, nExemplars=5)
        algorithm.entry(data_support)
    else:
        raise ValueError('No file dataset')


    # データの読み込み
    # input_image_path = '/home/datasets/VOCclassification/val/sheep/0_2008_002536.jpg'
    input_image_path = args_opt.image
    # input_image_path = '/home/datasets/VOCdetection/trainvalset_ID_15-19_images/2008_003147.jpg'
    input_name = input_image_path.split('/')[-1].split('.')[0]
    shutil.copy2(args_opt.image, "./output/GradCam/" + input_image_path.split('/')[-1])


    transforms_list = []
    transforms_list.append(transforms.Resize(84))
    transforms_list.append(transforms.CenterCrop(84))
    transforms_list.append(lambda x: np.asarray(x))
    transforms_list.append(transforms.ToTensor())
    mean_pix = [0.485, 0.456, 0.406]
    std_pix = [0.229, 0.224, 0.225]
    transforms_list.append(transforms.Normalize(mean=mean_pix, std=std_pix))
    img_transform = transforms.Compose(transforms_list)

    input = Image.open(input_image_path)
    # print(input.size)
    input = np.array(input)

    input = Image.fromarray(input)
    input = img_transform(input)
    input = input[np.newaxis,:,:,:]

    # 認識処理
    algorithm.classifer_loss_step(input,input_name)
