# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils

from pdb import set_trace as breakpoint
import time
from tqdm import tqdm

from . import Algorithm


def top1accuracy(output, target):
    _, pred = output.max(dim=1)
    pred = pred.view(-1)
    target = target.view(-1)
    accuracy = 100 * pred.eq(target).float().mean()
    return accuracy


def activate_dropout_units(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.training = True


class FewShot_new(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.nKbase = torch.LongTensor()
        self.activate_dropout = (
            opt['activate_dropout'] if ('activate_dropout' in opt) else False)
        self.keep_best_model_metric_name = 'AccuracyNovel'
        self.count = 0

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['images_train'] = torch.FloatTensor()
        self.tensors['labels_train'] = torch.LongTensor()
        self.tensors['labels_train_1hot'] = torch.FloatTensor()
        self.tensors['images_test'] = torch.FloatTensor()
        self.tensors['labels_test'] = torch.LongTensor()
        self.tensors['Kids'] = torch.LongTensor()

    def set_tensors(self, tloader_test, tloader_ttrain):
    # def set_tensors(self,batch, tloader_test, tloader_ttrain):
        train_test_stage = 'fewshot'
        #print("batch : ", batch)
        #print("tloader_test : ",tloader_test)
        images_train, labels_train = tloader_ttrain
        images_test, labels_test = tloader_test
        K = torch.LongTensor([[label for label in range(20)]])
        nKbase = torch.LongTensor([10])
        # images_train, labels_train, images_test, labels_test, K, nKbase = batch
        # print("images_train.size : ", images_train.size())  #(batch,トレーニングサンプル数,チャンネル、たて、よこ)
        # print("labels_train.size : ", labels_train.size())  #(batch,トレーニングサンプルの正解ラベル)
        # print("images_test.size : ", images_test.size())    #(batch,テストサンプル数,チャンネル、たて、よこ)
        # print("labels_test.size : ", labels_test.size())    #(batch,テストサンプルの正解ラベル)
        # print("K : ", K.size()) #ラベル名
        # print("nKbase : ", nKbase.size())   #ベースカテゴリの数

        self.nKbase = nKbase.squeeze().item()
        self.tensors['images_train'].resize_(images_train.size()).copy_(images_train)
        self.tensors['labels_train'].resize_(labels_train.size()).copy_(labels_train)
        labels_train = self.tensors['labels_train']

        nKnovel = 1 + labels_train.max() - self.nKbase

        labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
        labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
        self.tensors['labels_train_1hot'].resize_(labels_train_1hot_size).fill_(0).scatter_(
            len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.nKbase, 1)
        self.tensors['images_test'].resize_(images_test.size()).copy_(images_test)
        self.tensors['labels_test'].resize_(labels_test.size()).copy_(labels_test)
        self.tensors['Kids'].resize_(K.size()).copy_(K)

        return train_test_stage

    def test_step(self, tloader_test, tloader_ttrain, rect_count):
        self.rect_count = rect_count
        infer_label, smx_list = self.process_batch(tloader_test, tloader_ttrain, do_train=False)

        return infer_label, smx_list

    # def test_step(self,batch, tloader_test, tloader_ttrain):
    #     self.process_batch(batch,tloader_test, tloader_ttrain, do_train=False)

    def process_batch(self, tloader_test, tloader_ttrain, do_train):
        process_type = self.set_tensors(tloader_test, tloader_ttrain)

        if process_type=='fewshot':
            infer_label, smx_list = self.process_batch_fewshot_without_forgetting(do_train=do_train)
        else:
            raise ValueError('Unexpected process type {0}'.format(process_type))

        return infer_label, smx_list


    # def process_batch(self,batch, tloader_test, tloader_ttrain, do_train):
    #     process_type = self.set_tensors(batch,tloader_test, tloader_ttrain)
    #
    #     if process_type=='fewshot':
    #         self.process_batch_fewshot_without_forgetting(do_train=do_train)
    #     else:
    #         raise ValueError('Unexpected process type {0}'.format(process_type))

    def process_batch_fewshot_without_forgetting(self, do_train=True):
        images_train = self.tensors['images_train']
        labels_train = self.tensors['labels_train']
        labels_train_1hot = self.tensors['labels_train_1hot']
        images_test = self.tensors['images_test']
        labels_test = self.tensors['labels_test']
        Kids = self.tensors['Kids']
        nKbase = self.nKbase

        feat_model = self.networks['feat_model']
        classifier = self.networks['classifier']
        criterion = self.criterions['loss']

        feat_model.eval()


        #*********************** SET TORCH VARIABLES ***************************
        images_test_var = Variable(images_test)
        labels_test_var = Variable(labels_test, requires_grad=False)
        Kbase_var = (None if (nKbase==0) else
            Variable(Kids[:,:nKbase].contiguous(),requires_grad=False))
        labels_train_1hot_var = Variable(labels_train_1hot, requires_grad=False)
        images_train_var = Variable(images_train)
        #***********************************************************************

        #************************* FORWARD PHASE: ******************************

        #************ EXTRACT FEATURES FROM TRAIN & TEST IMAGES ****************
        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)
        features_train_var = feat_model(
            images_train_var.view(batch_size * num_train_examples, channels, height, width)
        )
        features_test_var = feat_model(
            images_test_var.view(batch_size * num_test_examples, channels, height, width)
        )
        features_train_var = features_train_var.view(
            [batch_size, num_train_examples,] + list(features_train_var.size()[1:])
        )
        features_test_var = features_test_var.view(
            [batch_size, num_test_examples,] + list(features_test_var.size()[1:])
        )
        # Make sure that no gradients are backproagated to the feature
        # extractor when the feature extraction model is freezed.
        features_train_var = Variable(features_train_var.data, volatile=False)
        features_test_var = Variable(features_test_var.data, volatile=False)
        #***********************************************************************

        #************************ APPLY CLASSIFIER *****************************
        if self.nKbase > 0:
            cls_scores_var = classifier(
                features_test=features_test_var,
                Kbase_ids=Kbase_var,
                features_train=features_train_var,
                labels_train=labels_train_1hot_var)
        else:
            cls_scores_var = classifier(
                features_test=features_test_var,
                features_train=features_train_var,
                labels_train=labels_train_1hot_var)

        cls_scores_var = cls_scores_var.view(batch_size * num_test_examples, -1) #(10,66)
        labels_test_var = labels_test_var.view(batch_size * num_test_examples)
        #***********************************************************************

        infer_label_prob, infer_label = cls_scores_var.data.max(dim=1)
        #print ("features_train_var   : ", features_train_var.data)
        #print ("features_train_var_size   : ", features_train_var.data.size())

        # max_acc = 0
        # max_id = 0
        smx_list = []
        # infer_label_prob, infer_label = sess.run(cls_scores_var.data.max(dim=1))
        for i in range(self.rect_count):
            smx = F.softmax(cls_scores_var.data[i],dim=0).cpu().numpy()
            np.set_printoptions(suppress=True, precision=3)
            # print ('infer_label_prob softmax : ', smx*100)
            smx_max = smx.max()
            # print ('infer_label_prob softmax max :', smx_max*100)
            smx_list.append(smx_max*100)
            # if max_acc < smx_max:
            #     max_acc = smx_max
            #     max_id = i

        infer_label = infer_label.cpu().numpy()
        # labels = labels_test_var.data.cpu().numpy()
        #print ('infer_label_prob softmax :',cls_scores_var.data)
        # print ("rect_count : ", self.rect_count)
        # print ("cls_max   : ", infer_label)
        # print ("label : ", labels)

        return infer_label, smx_list
