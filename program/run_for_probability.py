#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:52:44 2018

@author: tat2016

Get probability for an input
"""

from keras.models import load_model
import os
import data_generator
import h5py
import re
import numpy as np
import sys

import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model

plt.switch_backend('agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

ngpu = 4

batch_size = 24 * ngpu

segment_size = 4000
version = ''
nbr_feature = 10
rnn_len = 800
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len}




input_folder = sys.argv[1]
output_folder = sys.argv[2]


boundary_model_file = 'boundary_pred_4k.h5'

loop_model_file = 'loop_pred_4k_1gpu.h5' #'loop_pred_4k_1gpu.h5' #'loop_pred_4k_1gpu.h5' #'loop_pred_4k_1gpu_8936.h5'# 'loop_pred_4k_8919_train_gm12878_k562.h5' #'loop_pred_4k_8913_train_gm12878_k562.h5' #'loop_pred_4k_train_gm12878_k562.h5' #'loop_pred_4k_nonboundary_full_9133.h5'



def get_prediction(model_file, data_file, isloop=True):
    
    if isinstance(model_file, str):
        model = load_model(model_file)
    else:
        model = model_file
    
    rs = {}
    
    if isloop:
        generator = data_generator.DataGeneratorLoopSeq(data_file, None, shuffle = False, use_reverse=False, **params)
    else:
        generator = data_generator.DataGenerator(data_file, None, shuffle = False, use_reverse=False, **params)

    prediction = model.predict_generator(generator, verbose=1)
    #print(prediction, len(generator.ids_list))
    prediction = np.array(prediction).ravel()
    
    labels = generator.ids_list
    for i,k in enumerate(labels):
        if k in rs and abs(rs[k] - prediction[i]) > 0.00001:
            print('error in get_prediction, sample order is messed up')
        rs[k] = prediction[i]

    
    
    return(rs)
    
def output_prob(rs, file_name):
    fout = open(file_name,'w')
    for k,v in rs.items():
        fout.write('%s: \t %.10f\n' % (k,v))
      
    fout.close()
    

if not os.path.exists(output_folder):
    os.makedirs(output_folder)    


model = load_model(loop_model_file) 
model = multi_gpu_model(model, gpus = ngpu)


patient_files = os.listdir(input_folder)

for i in patient_files:
#    if not 'gm12878' in i:
#        continue
    
    input_file = os.path.join(input_folder, i)
    
    dataset = h5py.File(input_file, 'r')
    
    print(input_file)
    if len(dataset.items()) == 0:
        continue
    
    output_file = os.path.join(output_folder, i.replace('.mat','.txt'))
    
    rs = get_prediction(model, input_file)
    
    output_prob(rs, output_file)



