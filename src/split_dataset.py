# -*- coding: utf-8 -*-
from glob import glob
from os.path import join
import random
import os
import shutil
import logging
from tqdm import tqdm

dataset = "../../datasets/PASCAL3D+_release1.1/datasets"
dataset_root = os.path.join(dataset, 'root')
dataset_train = os.path.join(dataset, 'train')
dataset_trainval = os.path.join(dataset, 'trainval')
dataset_val = os.path.join(dataset, 'val')
dataset_test = os.path.join(dataset, 'test')

logger = logging.getLogger(__name__)

strHandler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
strHandler.setFormatter(formatter)
logger.addHandler(strHandler)
logger.setLevel(logging.INFO)


if not os.path.exists(dataset_train):
    os.makedirs(dataset_train)

if not os.path.exists(dataset_val):
    os.makedirs(dataset_val)

if not os.path.exists(dataset_test):
    os.makedirs(dataset_test)

if not os.path.exists(dataset_trainval):
    os.makedirs(dataset_trainval)

# category list 取得
files = os.listdir(dataset_root)
for i in range(len(files)):
    logger.info('split category : %s [ %d / %d ]' % (files[i], i+1, len(files)))
    file_list = os.listdir(os.path.join(dataset_root,files[i]))

    # root から trainval, testに分割 ############################################
    logger.info('root --> test, trainval')
    num_file_list = len(file_list)
    files10 = random.sample(file_list, int(num_file_list*0.1))
    files90 = random.sample(file_list, num_file_list - int(num_file_list*0.1))
    logger.info('test : %d, trainval : %d' % (len(files10), len(files90)))

    # test 作成
    logger.info('make test dataset')
    category_path = os.path.join(dataset_test, files[i])
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    for j in tqdm(range(len(files10))):
        image_path = os.path.join(dataset_root, files[i], files10[j])
        shutil.copy(image_path, category_path)

    # trainval 作成
    logger.info('make trainval dataset')
    category_path = os.path.join(dataset_trainval, files[i])
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    for j in tqdm(range(len(files90))):
        image_path = os.path.join(dataset_root, files[i], files90[j])
        shutil.copy(image_path, category_path)


    # trainval から　train, val に分割 ##########################################
    logger.info('trainval --> val, train')
    num_file_list = len(files90)
    files30 = random.sample(files90, int(num_file_list*0.3))
    files70 = random.sample(files90, num_file_list - int(num_file_list*0.3))
    logger.info('val : %d, train : %d' % (len(files30), len(files70)))

    # val 作成
    logger.info('make val dataset')
    category_path = os.path.join(dataset_val, files[i])
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    for j in tqdm(range(len(files30))):
        image_path = os.path.join(dataset_root, files[i], files30[j])
        shutil.copy(image_path, category_path)

    # train 作成
    logger.info('make train dataset')
    category_path = os.path.join(dataset_train, files[i])
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    for j in tqdm(range(len(files70))):
        image_path = os.path.join(dataset_root, files[i], files70[j])
        shutil.copy(image_path, category_path)

    logger.info('finish ------------------------------------------------------')
