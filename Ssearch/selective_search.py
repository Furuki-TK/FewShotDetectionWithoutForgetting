# -*- coding: utf-8 -*-
import skimage.data
import scipy.misc
import numpy as np
from PIL import Image

import os
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in','--input', dest = 'input', help =
                            "Image / Directory containing images to perform detection upon",
                            default = " ", type = str)

    return parser.parse_args()

def SS(data):

    data_name = data.split('/')[-1].split('.')[0]
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

    # print(candidates)

    output_txt_name = '/home/output/txt_SS/' + data_name + '.txt'

    # ff = open(output_txt_name, 'w')

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x,y,w,h)
        # draw rectangles on the original image
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        # txt = str(x) + " " + str(y) + " " + str(x+w) + " " + str (y+h) + "\n"
        # ff.write(txt)

    # ff.close()
    output = '/home/output/SS_feat/' + data.split('/')[-1]
    plt.savefig(output)


    # image_number = 0
    # for x, y, w, h in candidates:
    #     print(x,y,w,h)
    #      # draw original image
    #     fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10, 10))
    #     ax[0].imshow(img)
    #
    #     # draw rectangles on the original image
    #     rect = mpatches.Rectangle(
    #         (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    #     ax[0].add_patch(rect)
    #
    #     # trim image and draw
    #     img2 = img[y:y+h, x:x+w]
    #     ax[1].imshow(img2)
    #
    #     # reshape trimed image and draw
    #     img3 = skimage.transform.resize(img2, (64, 64))
    #     ax[2].imshow(img3)
    #
    #     plt.savefig('output/figure[%d].png' % image_number)
    #     image_number += 1


if __name__ == "__main__":
    args_opt = arg_parse()
    # loading image
    image_txt = args_opt.input

    SS(image_txt)


    # # dataset_path = 'datasets/VOCDetection/test'
    #
    # dataset_path = '../datasets/VOCDetection/all_images/JPEGImages'
    # f = open(image_txt, 'r')
    #
    # file_list = []
    # for s in f:
    #     file_list.append(os.path.join(dataset_path,s.replace('\n','')))
    #
    # f.close()
    # # test1=file_list[:500]
    #
    # for image_name in file_list:
    #     SS(image_name)
