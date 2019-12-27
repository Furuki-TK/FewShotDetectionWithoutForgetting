# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import imp

import algorithms as alg
from dataloader import QuaryData, VOCTrainSet

dataset_path = './datasets/VOCdetection'
voc_path ='VOCdevkit/VOC2012/JPEGImages'

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-txt','--txt_images', dest = 'image', help =
                            "text file with list of input images name",
                            default = " ", type = str)
    parser.add_argument('--config', type=str, required=True, default='',
        help='config file with parameters of the experiment.')
    parser.add_argument('-id','--testset_id',dest = 'id' ,type=int, default=0,
        help='id is the number from 0 to 7')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')

    return parser.parse_args()

if __name__ ==  '__main__':
    args_opt = arg_parse()

    exp_config_file = os.path.join('.', 'config', args_opt.config + '.py')
    exp_directory = os.path.join('.', 'experiments', 'selective_search', args_opt.config)
    exp_model_directory = os.path.join('.', 'experiments', args_opt.config)

    # Load the configuration params of the experiment
    print('Launching experiment: %s' % exp_config_file)
    config = imp.load_source("",exp_config_file).config
    config['thres'] = 30.0
    NN = '_' + str(config['thres']) + '%_test'
    config['img_dir'] = '/home/output/SS+NM/images'+ NN
    config['txt_dir'] = '/home/output/SS+NM/txt' + NN
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
    test=file_list[319:]
    test.append(file_list[0]) # 確認用

    test1=file_list[0:1000]
    test2=file_list[1000:2000]
    test3=file_list[2000:3000]
    test4=file_list[3000:4000]
    test5=file_list[4000:5000]
    test6=file_list[5000:]

    # 検出処理
    print("Detection.....")
    if args_opt.id == 0:
        for image_name in test:
            data_query = QuaryData(image_path=image_name)
            algorithm.SS_classifer_opt(data_query)
    if args_opt.id == 1:
        for image_name in test1:
            data_query = QuaryData(image_path=image_name)
            algorithm.SS_classifer_opt(data_query)
    if args_opt.id == 2:
        for image_name in test2:
            data_query = QuaryData(image_path=image_name)
            algorithm.SS_classifer_opt(data_query)
    if args_opt.id == 3:
        for image_name in test3:
            data_query = QuaryData(image_path=image_name)
            algorithm.SS_classifer_opt(data_query)
    if args_opt.id == 4:
        for image_name in test4:
            data_query = QuaryData(image_path=image_name)
            algorithm.SS_classifer_opt(data_query)
    if args_opt.id == 5:
        for image_name in test5:
            data_query = QuaryData(image_path=image_name)
            algorithm.SS_classifer_opt(data_query)
    if args_opt.id == 6:
        for image_name in test6:
            data_query = QuaryData(image_path=image_name)
            algorithm.SS_classifer_opt(data_query)
    if args_opt.id == 10:
        for image_name in file_list:
            data_query = QuaryData(image_path=image_name)
            algorithm.SS_classifer_opt(data_query)
