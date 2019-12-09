import shutil
import os

input_path = '../../datasets/PASCAL3D+_release1.1/PASCAL/VOCdevkit/VOC2012/JPEGImages'
output_path = 'datasets/VOCDetection/test'

f = open('test_VOCimage_name.txt', 'r')

for s in f:
    data_name=s.replace('\n','')
    input_image = os.path.join(input_path,data_name)
    output_image = os.path.join(output_path,data_name)
    shutil.copy2(input_image, output_image)

f.close()
