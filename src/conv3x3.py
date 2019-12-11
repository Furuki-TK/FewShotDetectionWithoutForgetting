# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import os
import cv2
import imp
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch


def SS(data, data_name, output_dir, out_image=False):

    img = np.array(Image.open(data))
    assert(img.all() != None)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 200:
            continue
        # distorted rects
        x, y, w, h = r['rect']

        if w == 0 or h == 0:
            continue

        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    output_txt_name = data_name + '.txt'
    ff = open(os.path.join(output_dir, output_txt_name), 'w')

    if out_image:
        out_image_dir = os.path.join('/home/output/SS_features', data_name)
        if not os.path.exists(out_image_dir):
            os.makedirs(out_image_dir)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        ax.imshow(img)
        for x, y, w, h in candidates:

            txt = str(x) + " " + str(y) + " " + str(x+w) + " " + str (y+h) + "\n"
            ff.write(txt)

            # draw rectangles on the original image
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        output = os.path.join(out_image_dir, data.split('/')[-1].split('.')[0] + ".png")
        plt.savefig(output)
        plt.close()

    else:
        for x, y, w, h in candidates:

            txt = str(x) + " " + str(y) + " " + str(x+w) + " " + str (y+h) + "\n"
            ff.write(txt)

    ff.close()


def main():
    vgg16 = models.vgg16(pretrained=True)

    # 全層のパラメータを固定
    for param in vgg16.parameters():
        param.requires_grad = False

    # print(vgg16)

    # 第一層だけ抽出
    first_layer = vgg16.features[0]


    # GPUを使えるなら使う
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        first_layer = first_layer.cuda()

    first_layer.eval()

    # print(first_layer)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    image_txt = "/home/datasets/VOCDetection/valset_c1-20.txt"
    dataset_path = '/home/datasets/VOCDetection/all_images/JPEGImages'
    f = open(image_txt, 'r')
    file_list = []
    for s in f:
        file_list.append(os.path.join(dataset_path,s.replace('\n','')))

    f.close()

    for image_path in tqdm(file_list, desc='image'):
        # image_path = "../datasets/VOCDetection/all_images/JPEGImages/2008_000093.jpg"
        image_name = image_path.split('/')[-1].split('.')[0]

        conv_size = 3
        output_dir = "/home/output/SS_txt_features/conv" + str(conv_size) + "x" + str(conv_size)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # print('load image : ' + image_name + ".png")

        img = Image.open(image_path)
        img_tensor = preprocess(img)
        img_tensor.unsqueeze_(0)

        # print('output features')

        if use_gpu:
            output = first_layer(Variable(img_tensor.cuda()))
        else:
            output = first_layer(Variable(img_tensor))

        # print(output.size())
        xx = np.squeeze(output.data.cpu().numpy())      # (w, h, c)

        output_feature_dir = '/home/output/features'
        if not os.path.exists(output_feature_dir):
            os.makedirs(output_feature_dir)

        # print('--> save features')

        for i in range(len(xx)):

            mask = xx[i]
            mask = mask - np.min(mask)

            if np.max(mask) != 0:
                mask = mask / np.max(mask)

            feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            output_name = os.path.join(output_feature_dir ,'feature_ch_'+str(i)+'.png')
            cv2.imwrite(output_name, feature_map)

            if i == 0:
                masks = mask
            else:
                masks += mask

        masks = masks / (i+1)

        feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * masks), cv2.COLORMAP_JET))
        output_name = os.path.join(output_feature_dir ,'feature_ch_mean.png')
        cv2.imwrite(output_name, feature_map)

        input_list = os.listdir(output_feature_dir)

        # print("Selective Search")

        for input_feature in tqdm(input_list, desc='Selective Search'):
            input_feature_path = os.path.join(output_feature_dir, input_feature)
            input_feature_name = input_feature.split(".")[0]
            output_SS_txt_dir = os.path.join(output_dir,input_feature_name)
            if not os.path.exists(output_SS_txt_dir):
                os.makedirs(output_SS_txt_dir)

            SS(input_feature_path, image_name, output_SS_txt_dir)


if __name__ == "__main__":
    main()
