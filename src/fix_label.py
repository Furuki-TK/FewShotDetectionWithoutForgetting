# -*- coding: utf-8 -*-
from __future__ import print_function

import os

txt_path = './output/txt_30%_c20a'
output_dir = './output/txt_30%_c20'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

txt_list = os.listdir(txt_path)

for txt_file in txt_list:
    f=open(os.path.join(txt_path,txt_file),'r')
    g=open(os.path.join(output_dir,txt_file),'w')
    for j in f:
        contxt = j
        if 'birds' in contxt:
            contxt= contxt.replace('birds','bird')
        elif 'people' in contxt:
            contxt=contxt.replace('people','person')
        g.write(contxt)
    f.close()
    g.close()
