# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import imp
import cv2
import skimage
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import algorithms as alg
from dataloader import MiniImageNet, FewShotDataloader, ImageNetTrainSet, OriginalTestSet, VOCTrainSet


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-itxt','--imagetxt', dest = 'image', help =
                            "Image / Directory containing images to perform detection upon",
                            default = " ", type = str)
    parser.add_argument('--config', type=str, required=True, default='',
        help='config file with parameters of the experiment. It is assumed that all'
             ' the config file is placed on  ')
    parser.add_argument('--evaluate', default=False, action='store_true',
        help='If True, then no training is performed and the model is only '
             'evaluated on the validation or test set of MiniImageNet.')
    parser.add_argument('--num_workers', type=int, default=0,
        help='number of data loading workers')
    parser.add_argument('-id','--testset_id',dest = 'id' ,type=int, default=0,
        help='testset_id')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--testset', default=False, action='store_true',
        help='If True, the model is evaluated on the test set of MiniImageNet. '
             'Otherwise, the validation set is used for evaluation.')

    return parser.parse_args()

if __name__ ==  '__main__':
    args_opt = arg_parse()

    exp_config_file = os.path.join('.', 'config', args_opt.config + '.py')
    exp_directory = os.path.join('.', 'experiments', args_opt.config)

    # Load the configuration params of the experiment
    print('Launching experiment: %s' % exp_config_file)
    config = imp.load_source("",exp_config_file).config
    config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
    print('Loading experiment %s from file: %s' %
          (args_opt.config, exp_config_file))
    print('Generated logs, snapshots, and model files will be stored on %s' %
          (config['exp_dir']))

    algorithm = alg.FewShot_new(config)

    print("Loading network.....")
    algorithm.load_checkpoint(epoch='*', train=False, suffix='.best')
    print("Network successfully loaded")

    if args_opt.cuda: # enable cuda
        algorithm.load_to_gpu()

    data_test_opt = config['data_test_opt']

    if 'ImageNet' in args_opt.config:
        print("Imagenet")
        tloader_ttrain = ImageNetTrainSet(phase='train',
                                        category=2, # number of novel categories.
                                        dpc=5,
                                        nExemplars=-1, # number of training examples per novel category.
                                        )
    elif 'VOC' in args_opt.config:
        print("VOC")
        tloader_ttrain = VOCTrainSet(phase='train',
                                        category=5, # number of novel categories.
                                        nExemplars=5
                                        )


    # image_name = args_opt.image
    # tloader_test = OriginalTestSet(data=image_name)
    # algorithm.tester(tloader_test, tloader_ttrain)
    #
    image_txt = args_opt.image
    dataset_path = './datasets/VOCDetection/all_images/JPEGImages'
    f = open(image_txt, 'r')

    file_list = []
    for s in f:
        file_list.append(os.path.join(dataset_path,s.replace('\n','')))

    f.close()
    # test1=file_list[:500]
    test2=file_list[0:1000]
    # test3=file_list[1000:1500]
    test4=file_list[1000:2000]
    # test5=file_list[2000:2500]
    test6=file_list[2000:3000]
    # test7=file_list[3000:3500]
    test8=file_list[3000:4000]
    # test9=file_list[4000:4500]
    test10=file_list[4000:5000]
    test12=file_list[5000:]

    if args_opt.id ==0:
        for image_name in file_list:
            tloader_test = OriginalTestSet(data=image_name)
            algorithm.tester(tloader_test, tloader_ttrain, image_name)
    elif args_opt.id ==1:
        for image_name in test2:
            tloader_test = OriginalTestSet(data=image_name)
            algorithm.tester(tloader_test, tloader_ttrain, image_name)
    elif args_opt.id ==2:
        for image_name in test4:
            tloader_test = OriginalTestSet(data=image_name)
            algorithm.tester(tloader_test, tloader_ttrain, image_name)
    elif args_opt.id ==3:
        for image_name in test6:
            tloader_test = OriginalTestSet(data=image_name)
            algorithm.tester(tloader_test, tloader_ttrain, image_name)
    elif args_opt.id ==4:
        for image_name in test8:
            tloader_test = OriginalTestSet(data=image_name)
            algorithm.tester(tloader_test, tloader_ttrain, image_name)
    elif args_opt.id ==5:
        for image_name in test10:
            tloader_test = OriginalTestSet(data=image_name)
            algorithm.tester(tloader_test, tloader_ttrain, image_name)
    elif args_opt.id ==6:
        for image_name in test12:
            tloader_test = OriginalTestSet(data=image_name)
            algorithm.tester(tloader_test, tloader_ttrain, image_name)
            # if int(image_name.split('/')[-1].split('.')[0].split('_')[0])==2011 and int(image_name.split('/')[-1].split('.')[0].split('_')[-1]) > 1669:
            #     tloader_test = OriginalTestSet(data=image_name)
            #     algorithm.tester(tloader_test, tloader_ttrain, image_name)
