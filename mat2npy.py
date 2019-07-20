#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:20:27 2019

@author: xingyu
"""

import numpy as np
import h5py 
import json

# f = h5py.File('/Users/xingyu/Downloads/imdb-charades-ft-rgb.mat','r')
f = h5py.File('/home/jin/Downloads/imdb-charades-ft-rgb.mat','r')

num_frames = f['videos/frames']
num_frames = num_frames.value.squeeze()
length = num_frames.shape[0]

names = f['videos/name']
data = f['videos/data']

dict_num_frames = {}

for i in range(length):
    name = ''.join(chr(x) for x in f[names[0,i]].value.squeeze())
#    feature = np.expand_dims(np.expand_dims(f[data[0,i]].value, axis=-1), axis=-1)
    feature = np.expand_dims(f[data[0,i]].value, axis=-1)
    np.save('i3d_rgb_hakan2/'+name, feature)
#    num_frame = f[data[0,i]].value.shape[1]
#    dict_num_frames[name] = num_frame
    
    print(i)
    
#with open('num_frames_' + 'rgb' + '_' + 'hakan' + '.json', 'w') as outfile:# 9848-1
#    json.dump(dict_num_frames, outfile)
