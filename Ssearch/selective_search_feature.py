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


def SS(img, data_name, output_dir, out_image=False):
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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        ax.imshow(img)
        for x, y, w, h in candidates:

            txt = str(x) + " " + str(y) + " " + str(x+w) + " " + str (y+h) + "\n"
            ff.write(txt)

            # draw rectangles on the original image
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        output = os.path.join(out_image_dir, data_name + ".png")
        plt.savefig(output)

    else:
        for x, y, w, h in candidates:

            txt = str(x) + " " + str(y) + " " + str(x+w) + " " + str (y+h) + "\n"
            ff.write(txt)

    ff.close()

if __name__ == "__main__":
    args = arg_parse()
    # loading image
    image_list = os.listdir(args.input)

    output_dir = '/home/output/SS_feature_conv3x3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image in image_list:
        image = os.path.join(args.input, image)
        image_name = image.split('/')[-1].split('.')[0]
        img = np.array(Image.open(image))
        assert(img.all() != None)

        SS(image, image_name)
