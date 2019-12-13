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



class FewShotClassification(Algorithm):
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


    def set_tensors(self, process_type, data):
        if process_type == 'entry':
            images_train, labels_train = data
            K = torch.LongTensor([[label for label in range(20)]])
            nKbase = torch.LongTensor([10])

            self.nKbase = nKbase.squeeze().item()
            self.tensors['images_train'].resize_(images_train.size()).copy_(images_train)
            self.tensors['labels_train'].resize_(labels_train.size()).copy_(labels_train)
            labels_train = self.tensors['labels_train']

            nKnovel = 1 + labels_train.max() - self.nKbase

            labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            self.tensors['labels_train_1hot'].resize_(labels_train_1hot_size).fill_(0).scatter_(
                len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.nKbase, 1)
            self.tensors['Kids'].resize_(K.size()).copy_(K)

        elif process_type == 'classifer':
            images_test, labels_test = data
            self.tensors['images_test'].resize_(images_test.size()).copy_(images_test)
            self.tensors['labels_test'].resize_(labels_test.size()).copy_(labels_test)

        else:
            raise ValueError('Unexpected process type {0}'.format(process_type))


    def entry_step(self, data_support):
        process_type = 'entry'
        self.set_tensors(process_type, data_support)
        self.process_entry()

    def classifer_step(self, data_query, rect_count):
        process_type = 'classifer'
        self.rect_count = rect_count
        self.set_tensors(process_type, data_query)
        infer_label, smx_list = self.process_fewshot_without_forgetting()

        return infer_label, smx_list

    def process_entry(self, data_support):
        images_train = self.tensors['images_train']
        labels_train = self.tensors['labels_train']
        labels_train_1hot = self.tensors['labels_train_1hot']
        Kids = self.tensors['Kids']
        nKbase = self.nKbase

        self.feat_model = self.networks['feat_model']
        self.classifier = self.networks['classifier']

        self.feat_model.eval()

        #*********************** SET TORCH VARIABLES ***************************
        labels_train_1hot_var = Variable(labels_train_1hot, requires_grad=False)
        images_train_var = Variable(images_train)
        Kbase_var = (None if (nKbase==0) else
            Variable(Kids[:,:nKbase].contiguous(),requires_grad=False))

        batch_size, num_train_examples, channels, height, width = images_train.size()
        features_train_var = self.feat_model(
            images_train_var.view(batch_size * num_train_examples, channels, height, width)
        )
        features_train_var = features_train_var.view(
            [batch_size, num_train_examples,] + list(features_train_var.size()[1:])
        )

        features_train_var = Variable(features_train_var.data, volatile=False)

        #************************ APPLY CLASSIFIER *****************************
        self.classifier.get_classification_weights(
            Kbase_ids=Kbase_var,
            features_train=features_train_var,
            labels_train=labels_train_1hot_var)

    def process_fewshot_without_forgetting(self):
        images_test = self.tensors['images_test']
        labels_test = self.tensors['labels_test']

        self.feat_model.eval()

        #*********************** SET TORCH VARIABLES ***************************
        images_test_var = Variable(images_test)
        labels_test_var = Variable(labels_test, requires_grad=False)

        #************************* FORWARD PHASE: ******************************
        #************ EXTRACT FEATURES FROM TRAIN & TEST IMAGES ****************
        batch_size, num_test_examples, channels, height, width = images_test.size()
        features_test_var = self.feat_model(
            images_test_var.view(batch_size * num_test_examples, channels, height, width)
        )
        features_test_var = features_test_var.view(
            [batch_size, num_test_examples,] + list(features_test_var.size()[1:])
        )

        features_test_var = Variable(features_test_var.data, volatile=False)

        #************************ APPLY CLASSIFIER *****************************
        cls_scores_var = self.classifier(
            features_test=features_test_var)

        cls_scores_var = cls_scores_var.view(batch_size * num_test_examples, -1) #(10,66)
        infer_label_prob, infer_label = cls_scores_var.data.max(dim=1)

        smx_list = []

        for i in range(self.rect_count):
            smx = F.softmax(cls_scores_var.data[i],dim=0).cpu().numpy()
            np.set_printoptions(suppress=True, precision=3)
            smx_max = smx.max()
            smx_list.append(smx_max*100)

        infer_label = infer_label.cpu().numpy()

        return infer_label, smx_list
