# -*- coding: utf-8 -*-
import argparse
import os
import imp

import algorithms as alg

import shutil
import urllib
import pickle
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataloader import QuaryVOC, VOCTrainSet

dataset_path = './datasets/VOCdetection'
voc_path ='VOCdevkit/VOC2012/JPEGImages'

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest = 'image', help =
                            "text file with list of input images name",
                            default = " ", type = str)
    parser.add_argument('--config', type=str, default='miniImageNet_Conv128CosineClassifierGenWeightAttN5',
        help='config file with parameters of the experiment.')
    parser.add_argument('--id',dest = 'id' ,type=int, default=0,
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
    # config['thres'] = 30.0
    # NN = '_' + str(config['thres']) + '%_test'
    config['img_dir'] = '/home/output/GradCam_detection/images'
    config['txt_dir'] = '/home/output/GradCam_detection/txt'
    config['cam_dir'] = '/home/output/GradCam_detection/cam'
    config['exp_dir'] = exp_directory
    config['model_dir'] = exp_model_directory
    config['networks']['classifier']['def_file'] = 'architectures/ClassifierWithFewShotGenerationModule_new.py'
    if 'mini' in args_opt.config:
        config['name'] = 'mini'
    elif 'voc' in args_opt.config:
        config['name'] = 'voc'

    if not os.path.exists(config['img_dir']):
        os.makedirs(config['img_dir'])

    if not os.path.exists(config['txt_dir']):
        os.makedirs(config['txt_dir'])

    if not os.path.exists(config['cam_dir']):
        os.makedirs(config['cam_dir'])

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
    data_support = VOCTrainSet(phase='train',base=config['name'], category=5, nExemplars=5)
    algorithm.entry(data_support)

    # データの読み込み
    image_txt = args_opt.image
    if 'test' in image_txt.split('/')[-1]:
        dataset_path = os.path.join(dataset_path, 'imageset_test', voc_path)
    else:
        dataset_path = os.path.join(dataset_path, 'imageset_trainval', voc_path)

    f = open(image_txt, 'r')
    file_list = []
    for s in f:
        file_list.append(os.path.join(dataset_path,s.replace('\n','.jpg')))
    f.close()

    # データの分割
    test=file_list[0:3]
    # test.append(file_list[0]) # 確認用

    test1=file_list[0:1000]
    test2=file_list[1000:2000]
    test3=file_list[2000:3000]
    test4=file_list[3000:4000]
    test5=file_list[4000:5000]
    test6=file_list[5000:]

    # 特定の画像指定
    # input_image_path = args_opt.image
    # input_name = input_image_path.split('/')[-1].split('.')[0]
    # # shutil.copy2(args_opt.image, "/home/output/GradCam_detection/" + input_image_path.split('/')[-1])
    #
    # transforms_list = []
    # transforms_list.append(transforms.Resize(84))
    # transforms_list.append(transforms.CenterCrop(84))
    # transforms_list.append(lambda x: np.asarray(x))
    # transforms_list.append(transforms.ToTensor())
    # mean_pix = [0.485, 0.456, 0.406]
    # std_pix = [0.229, 0.224, 0.225]
    # transforms_list.append(transforms.Normalize(mean=mean_pix, std=std_pix))
    # img_transform = transforms.Compose(transforms_list)
    #
    # input = Image.open(input_image_path)
    # # print(input.size)
    # input = np.array(input)
    # input_origin = input
    # input_size = input.shape
    # input = Image.fromarray(input)
    # input = img_transform(input)
    # input = input[np.newaxis,:,:,:]

    # 検出処理
    print("Detection.....")
    if args_opt.id == 0:
        for i, image_name in enumerate(tqdm(test)):
            data_query = QuaryVOC(image_path=image_name)
            algorithm.GradCam_SS_opt(data_query)
    if args_opt.id == 1:
        for i, image_name in enumerate(tqdm(test1)):
            data_query = QuaryVOC(image_path=image_name)
            algorithm.GradCam_SS_opt(data_query)
    if args_opt.id == 2:
        for i, image_name in enumerate(tqdm(test2)):
            data_query = QuaryVOC(image_path=image_name)
            algorithm.GradCam_SS_opt(data_query)
    if args_opt.id == 3:
        for i, image_name in enumerate(tqdm(test3)):
            data_query = QuaryVOC(image_path=image_name)
            algorithm.GradCam_SS_opt(data_query)
    if args_opt.id == 4:
        for i, image_name in enumerate(tqdm(test4)):
            data_query = QuaryVOC(image_path=image_name)
            algorithm.GradCam_SS_opt(data_query)
    if args_opt.id == 5:
        for i, image_name in enumerate(tqdm(test5)):
            data_query = QuaryVOC(image_path=image_name)
            algorithm.GradCam_SS_opt(data_query)
    if args_opt.id == 6:
        for i, image_name in enumerate(tqdm(test6)):
            data_query = QuaryVOC(image_path=image_name)
            algorithm.GradCam_SS_opt(data_query)
    if args_opt.id == 10:
        for image_name in file_list:
            data_query = QuaryVOC(image_path=image_name)
            algorithm.GradCam_SS_opt(data_query)
