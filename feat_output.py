from __future__ import print_function
import argparse
import os
import imp
import cv2
import numpy as np
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

# from dataloader import MiniImageNet, FewShotDataloader
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--config', type=str, required=True, default='',
#     help='config file with parameters of the experiment. It is assumed that all'
#          ' the config file is placed on  ')
#
# args_opt = parser.parse_args()

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
	return out

class ConvNet(nn.Module):
    def __init__(self, opt):
        super(ConvNet, self).__init__()
        self.in_planes  = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        results = []
        for ii,model in enumerate(self.conv_blocks):
            for iii,layer in enumerate(model.layers):
                x = layer(x)
                if (ii < 3 and iii == 2) or (ii == 3 and iii == 1):
                    results.append(x)

        return results

def create_model(opt):
    return ConvNet(opt)

# exp_directory = os.path.join('.', 'experiments', args_opt.config)
# pretrained_path = '/home/experiments/pascalVOC_Conv128CosineClassifier/feat_model_net_epoch11.best'
pretrained_path = None
feat_model_opt = {'userelu': False, 'in_planes':3, 'out_planes':[10,10,128,128], 'num_stages':4}

def init_network():

    network = create_model(feat_model_opt)
    if pretrained_path != None:
        load_pretrained(network, pretrained_path)

    return network

def load_pretrained(network, pretrained_path):
    assert(os.path.isfile(pretrained_path))
    all_possible_files = glob.glob(pretrained_path)
    if len(all_possible_files) == 0:
        raise ValueError('{0}: no such file'.format(pretrained_path))
    else:
        pretrained_path = all_possible_files[-1]

    pretrained_model = torch.load(pretrained_path)
    if pretrained_model['network'].keys() == network.state_dict().keys():
        network.load_state_dict(pretrained_model['network'])
    else:
        print('==> WARNING: network parameters in pre-trained file'
                         ' %s do not strictly match' % (pretrained_path))
        for pname, param in network.named_parameters():
            if pname in pretrained_model['network']:
                print('==> Copying parameter %s from file %s' %
                                 (pname, pretrained_path))
                param.data.copy_(pretrained_model['network'][pname])

if __name__ == '__main__':
    feat_model = init_network()

    data = './datasets/VOCDetection/trainvalset_ID_15-19_images/2008_000093.jpg'

    transforms_list = []
    transforms_list.append(transforms.Resize(84))
    transforms_list.append(transforms.CenterCrop(84))
    transforms_list.append(lambda x: np.asarray(x))
    transforms_list.append(transforms.ToTensor())
    mean_pix = [0.485, 0.456, 0.406]
    std_pix = [0.229, 0.224, 0.225]
    transforms_list.append(transforms.Normalize(mean=mean_pix, std=std_pix))
    img_transform = transforms.Compose(transforms_list)

    input = np.array(Image.open(data))
    input = Image.fromarray(input)
    input = img_transform(input)
    input = input[np.newaxis,:,:,:]

    # input = torch.randn(1, 3, 84, 84)
    out = feat_model(input)

    count = 0
    for x in out:
        count += 1
        xx = np.squeeze(x.data.cpu().numpy())
        print(len(xx[0]))
        for i in range(len(xx)):
            mask = cv2.resize(xx[i], (84,84))
            mask = mask - np.min(mask)

            if np.max(mask) != 0:
                mask = mask / np.max(mask)

            feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            # feature_map = np.float32(np.uint8(255 * mask))
            output_name = 'output/feat_result/feature_Conv'+str(count) + '_' +str(i) +'.png'
            cv2.imwrite(output_name,feature_map)
