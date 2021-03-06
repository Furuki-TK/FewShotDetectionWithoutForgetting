# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import utils
import scipy.optimize as optimize

from pdb import set_trace as breakpoint
import time
from tqdm import tqdm
from PIL import Image
import cv2
from . import Algorithm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import selectivesearch


class FewShotClassification(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.nKbase = torch.LongTensor()
        self.activate_dropout = (
            opt['activate_dropout'] if ('activate_dropout' in opt) else False)
        self.keep_best_model_metric_name = 'AccuracyNovel'
        self.count = 0 #image count (num is image ID)
        self.build_label_id(opt['name'])

    def build_label_id(self,opt_name):
        if opt_name == 'mini':
            self.boost = 64
        elif opt_name == 'voc':
            self.boost = 0

        f = open('id2label.txt','r')
        buf = f.read().split()
        self.label2id = {}
        for n in buf:
            id,label_name = n.split(',')
            self.label2id[label_name] = int(id)+self.boost
        f.close()

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
            K = torch.LongTensor([[label for label in range(20+self.boost)]])
            nKbase = torch.LongTensor([self.opt['data_test_opt']['nKbase']])

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
            self.tensors['images_test'].resize_((1,1,3,84,84))


        else:
            raise ValueError('Unexpected process type {0}'.format(process_type))


    def entry_step(self, data_support):
        process_type = 'entry'
        self.set_tensors(process_type, data_support)
        self.process_entry(data_support)

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

    def process_classifer(self, param):
        x, y, w, h = map(int, param)

        w_max = len(self.data_query.data[0])
        h_max = len(self.data_query.data)
        if x > w_max:
            x = w_max - w
        if x < 0:
            x = 0
        if x+w > w_max:
            w = w_max - x
        if x+w < 0:
            w = 0 - x
        if y > h_max:
            y = h_max - h
        if y < 0:
            y = 0
        if y+h > h_max:
            h = h_max - y
        if y+h < 0:
            h = 0 - y
        if w <= 0:
            if x == w_max:
                x = w_max - 1
                w = 1
            else:
                w = 1
        if h <= 0:
            if y == h_max:
                y = h_max - 1
                h = 1
            else:
                h = 1

        # print(x, y, w, h)
        test_data = self.data_query.data[y:y+h, x:x+w]
        test_data = Image.fromarray(test_data)
        test_data = self.data_query.transform(test_data)
        test_data = torch.stack([test_data], dim=0)
        test_data = torch.stack([test_data], dim=0)

        return self.process_fewshot_classifer_without_forgetting(test_data)

    def classifer_step(self, query, data_query):
        process_type = 'classifer'
        self.data_query = data_query
        self.rect_count = data_query.rect_count
        self.set_tensors(process_type, query)
        smx_list = []
        infer_label = []
        for i in tqdm(range(self.rect_count)):
            param = self.data_query.rect_xywh[i]
            results = optimize.minimize(self.process_classifer, x0=param, method='Nelder-Mead')
            self.data_query.rect_xywh[i] = map(int,results.x)
            smx_list.append(self.infer_prob)
            infer_label.append(self.infer_label)

        return infer_label, smx_list

    def process_fewshot_classifer_without_forgetting(self, query):
        images_test = self.tensors['images_test'].copy_(query)

        self.feat_model.eval()

        #*********************** SET TORCH VARIABLES ***************************
        images_test_var = Variable(images_test)

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
        cls_scores_var = cls_scores_var.view(batch_size * num_test_examples, -1)

        smx = F.softmax(cls_scores_var.data, dim=1).cpu().numpy()
        self.infer_prob = smx.max()*100
        self.infer_label = smx.argmax()

        return 1.0/self.infer_prob*100

    def classifer_loss_step(self, query, query_name):
        process_type = 'classifer'
        self.set_tensors(process_type, query)
        maps, map = self.process_classifer_and_loss(query)

        maps = transforms.ToPILImage()(maps[0])

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        ax.imshow(maps.resize((224,224)))
        name = "./output/GradCam/" + 'GradCAM_' + query_name
        plt.savefig(name)

    def process_classifer_and_loss(self, query):
        images_test = self.tensors['images_test'].copy_(query)
        criterion  = self.criterions['loss']

        self.feat_model.eval()

        images_test_var = Variable(images_test)

        batch_size, num_test_examples, channels, height, width = images_test.size()
        features_test_var = self.feat_model(
            images_test_var.view(batch_size * num_test_examples, channels, height, width)
        )
        target_features = features_test_var
        features_test_var = features_test_var.view(
            [batch_size, num_test_examples,] + list(features_test_var.size()[1:])
        )
        # print('size',features_test_var.size())
        # gh = features_test_var.register_hook(self.save_gradient)
        features_test_var = Variable(features_test_var.data, volatile=False)


        #************************ APPLY CLASSIFIER *****************************
        cls_scores_var, cls_weights = self.classifier(
            features_test=features_test_var)

        cls_scores_var = cls_scores_var.view(batch_size * num_test_examples, -1)
        # print('min',features_test_var.data.min(dim=1))
        print('output : ',cls_scores_var)

        smx = F.softmax(cls_scores_var.data, dim=1).cpu().numpy()
        par = [[100.0]*74]
        out = smx*par
        print('softmax', out)

        infer_label_prob, infer_label_id = cls_scores_var.data.max(dim=1)
        print('pred No1 id :',infer_label_id,' pred No1 dist :',infer_label_prob)


        # label_id = infer_label_id
        cid = 73    #可視化したいクラスのID
        print('id :',cid,' pred dist :',cls_scores_var[0][cid])
        print('id :',cid,' pred prob :',out[0][cid])
        label_id = torch.cuda.LongTensor([cid])
        target_weights = cls_weights[0][label_id]
        # print('size :',target_weights.size())
        weight = torch.cuda.FloatTensor()
        weight = weight.resize_(128,5,5)
        for i in range(128):
            for j in range(5):
                for k in range(5):
                    weight[i][j][k] = target_weights[0][i*25:(i+1)*25].mean()

        # print('target_weights : ',target_weights[0][0*25:1*25])
        # print('target_weights : ',target_weights[0][1*25:2*25])
        # print('target_weights mean : ',target_weights[0][0*25:1*25].mean())
        # print('target_weights : ',target_weights[0][1*25:2*25].mean())
        # print(weight[1])
        target_features = target_features.view(128, 5, 5)
        # print('target_features',target_features.size())
        # print('weight',weight.size())

        mask = F.relu((weight * target_features).sum(dim=0)).squeeze(0)

        # print('mask', mask.size())

        feature_maps = []
        mask = cv2.resize(mask.data.cpu().numpy(), (84,84))
        mask = mask - np.min(mask)

        if np.max(mask) != 0:
            mask = mask / np.max(mask)

        query = query.view(3, 84, 84)
        feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
        cam = feature_map + np.float32((np.uint8(np.transpose(query,(1,2,0))*255)))
        cam = cam - np.min(cam)

        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        feature_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))

        return feature_maps, feature_map

    def gradcam_and_detection_step(self, query, data_query):
        process_type = 'classifer'
        self.set_tensors(process_type, query)
        cls_scores_var, smx_par, target_features, cls_weights = self.process_classifer_and_output_feature(query)

        cam_list = {}
        bbox_list = []
        for label in data_query.label_list:
            cid = self.label2id[label]

            maps = self.grad_cam(target_features, cid, cls_weights, query)
            attention = transforms.ToPILImage()(maps[0])

            if data_query.input_size[1] < data_query.input_size[0]:
                attention = attention.resize((data_query.input_size[0],data_query.input_size[0]))
                size_p = 'w'
            else:
                attention = attention.resize((data_query.input_size[1],data_query.input_size[1]))
                size_p = 'h'

            attention = np.array(attention)
            candidates = self.Ssearch(attention)
            for x, y, w, h in candidates:
                if size_p == 'h':
                    y = y*data_query.input_size[0]/data_query.input_size[1]
                    h = h*data_query.input_size[0]/data_query.input_size[1]
                else:
                    x = x*data_query.input_size[1]/data_query.input_size[0]
                    w = w*data_query.input_size[1]/data_query.input_size[0]

                txt = label + " " + str(smx_par[0][cid]) + " " + str(x) + " " + str(y) + " " + str(x+w) + " " + str (y+h) + "\n"
                bbox_list.append(txt)

            cam = transforms.ToPILImage()(maps[1])
            cam = cam.resize((data_query.input_size[1],data_query.input_size[0]))
            cam_list[label] = cam

            # print('id :',cid,' pred dist :',cls_scores_var[0][cid])
            # print('id :',cid,' pred prob :',smx_par[0][cid])

        return bbox_list, cam_list

    def process_classifer_and_output_feature(self, query):
        images_test = self.tensors['images_test'].copy_(query[0])
        criterion  = self.criterions['loss']

        self.feat_model.eval()

        #************************ OUTPUT FEATURES *****************************
        images_test_var = Variable(images_test)
        batch_size, num_test_examples, channels, height, width = images_test.size()
        features_test_var = self.feat_model(
            images_test_var.view(batch_size * num_test_examples, channels, height, width)
        )

        target_features = features_test_var

        #************************ APPLY CLASSIFIER *****************************
        features_test_var = features_test_var.view(
            [batch_size, num_test_examples,] + list(features_test_var.size()[1:])
        )
        features_test_var = Variable(features_test_var.data, volatile=False)
        cls_scores_var, cls_weights = self.classifier(
            features_test=features_test_var)

        cls_scores_var = cls_scores_var.view(batch_size * num_test_examples, -1)

        smx = F.softmax(cls_scores_var.data, dim=1).cpu().numpy()
        par = [[100.0]*84]
        smx_par = smx*par

        # np.set_printoptions(precision=3)
        # print('cos_dist : ', cls_scores_var)
        # print('softmax', smx_par)
        #
        # infer_label_prob, infer_label_id = cls_scores_var.data.max(dim=1)
        # print('pred No1 id :',infer_label_id,' pred No1 dist :',infer_label_prob)

        return cls_scores_var, smx_par, target_features, cls_weights

    def grad_cam(self, target_features, cid, cls_weights, query):
        label_id = torch.cuda.LongTensor([cid])
        target_weights = cls_weights[0][label_id]
        # print('size :',target_weights.size())
        weight = torch.cuda.FloatTensor()
        weight = weight.resize_(128,5,5)
        for i in range(128):
            for j in range(5):
                for k in range(5):
                    weight[i][j][k] = target_weights[0][i*25:(i+1)*25].mean()

        # print('target_weights : ',target_weights[0][0*25:1*25])
        # print('target_weights : ',target_weights[0][1*25:2*25])
        # print('target_weights mean : ',target_weights[0][0*25:1*25].mean())
        # print('target_weights : ',target_weights[0][1*25:2*25].mean())
        # print(weight[1])
        target_features = target_features.view(128, 5, 5)
        # print('target_features',target_features.size())
        # print('weight',weight.size())

        mask = F.relu((weight * target_features).sum(dim=0)).squeeze(0)

        # print('mask', mask.size())
        feature_maps = []
        mask = cv2.resize(mask.data.cpu().numpy(), (84,84))
        mask = mask - np.min(mask)

        if np.max(mask) != 0:
            mask = mask / np.max(mask)

        feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))

        for i in range(84):
            for j in range(84):
                if mask[i][j] <= 0.67:
                    mask[i][j] = 0.0
                else:
                    mask[i][j] = 1.0

        feature_map_mod = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
        attention = feature_map_mod - np.min(feature_map_mod)

        if np.max(attention) != 0:
            attention = attention / np.max(attention)

        feature_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * attention), cv2.COLOR_BGR2RGB)))

        query[0] = query[0].view(3, 84, 84)
        cam = feature_map + np.float32((np.uint8(np.transpose(query[0],(1,2,0))*255)))
        cam = cam - np.min(cam)

        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        feature_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))

        return feature_maps

    def Ssearch(self, image):
        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(
            image, scale=500, sigma=0.9, min_size=10)

        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 200 pixels
            if r['size'] < 200:
                continue
            #
            # if r['size'] > 5000:
            #     continue
            # distorted rects

            x, y, w, h = r['rect']

            if w == 0 or h == 0:
                continue

            if w / h > 1.2 or h / w > 1.2:
                continue
            candidates.add(r['rect'])

        return candidates
