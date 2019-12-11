#!/bin/sh

python IoU.py -det output/SS_txt_features/conv3x3/feature_ch_ -gt datasets/VOCDetection/grandtruth_txt -ch 0

for var in `seq 63`  #範囲の書き方(Bash独自) => {0..4}
do
  python IoU.py -det output/SS_txt_features/conv3x3/feature_ch_ -gt datasets/VOCDetection/grandtruth_txt -ch $var
done

python IoU.py -det output/SS_txt_features/conv3x3/feature_ch_ -gt datasets/VOCDetection/grandtruth_txt -ch mean
