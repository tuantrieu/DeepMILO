#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:16:54 2018

@author: tat2016

Loop model to predict loop from sequence only

Note: when training on multiple GPU, loading into a single GPU results in error. 
Try to save the single-GPU model (before it is copied to multiple GPUs), they shares the same weights
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"


import matplotlib.pyplot as plt
plt.switch_backend('agg')


from keras.models import Model, load_model
from keras.layers import Dense, Dropout, BatchNormalization,\
                        Input, concatenate, Subtract

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU

from keras.utils import multi_gpu_model

import data_generator
import roc_callback

from datetime import datetime

import common_function_evaluation


################ Input
    
segment_size = 4000
epoch = 20

batch_size = 60


log_file = "log/loop_prediction_log_%s.txt" % (datetime.now().strftime('%H_%M_%d_%m_%Y'))

cna = 2 #copy number alteration

leaky = 0.2
dropout = 0.00

rnn_len = 800 #sequence length to run RNN, set to 0 to disable using RNN


version = ''




existing_model_file = '' #'loop_pred_4k_best_weight.h5'


boundary_model_file =  'boundary_pred_4k_cnnlstm800_1gpu.h5'#'boundary_pred_4k_1gpu.h5'
boundary_direction_file = 'direction_pred_4k_1gpu.h5' #'direction_pred_4k_9642_train_gm12878_k562.h5' #'direction_pred_4k_9714.h5'



output_model_cnn = "loop_pred_%dk%s.h5" % (segment_size/1000, version)
output_best_model = "loop_pred_%dk_best%s.h5" % (segment_size/1000, version)




file_train_data = 'data_loop_4k_train.mat'
file_train_label = 'label_loop_4k_train.mat'

file_val_data = 'data_loop_4k_val.mat'
file_val_label = 'label_loop_4k_val.mat'

file_test_data = 'data_nonloop_4knegConv.mat'
file_test_label = 'label_nonloop_4knegConv.mat'


all_letters = 'ACGTN'
n_letters = len(all_letters)

nbr_feature = n_letters * 2# number of features

flog = open(log_file,"w")


##########
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len}

train_generator = data_generator.DataGeneratorLoopSeq(file_train_data, file_train_label, shuffle = True, **params)
val_generator = data_generator.DataGeneratorLoopSeq(file_val_data, file_val_label, shuffle = False, **params) #set shuffle=False to calculate AUC
test_generator = data_generator.DataGeneratorLoopSeq(file_test_data, file_test_label, shuffle = False, use_reverse = False, **params) #set shuffle=False to calculate AUC


################################################
'''Boundary 1'''
seq_input1 = Input(shape=(segment_size,nbr_feature,1))
z11_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn11')
z12_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn12')

'''Boundary 2'''
seq_input2 = Input(shape=(segment_size,nbr_feature,1))
z21_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn21')
z22_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn22')

################################################
boundary_model_tmp = load_model(boundary_model_file)


boundary_model_tmp.layers[5].name = 'boundary_cat'

'''running through CNN and RNN(LSTM) in the boundary model'''
c1 = boundary_model_tmp.layers[3](seq_input1)
z11 = boundary_model_tmp.layers[4](z11_input)
z12 = boundary_model_tmp.layers[4](z12_input)
x1 = boundary_model_tmp.layers[5]([c1,z11,z12])


c2 = boundary_model_tmp.layers[3](seq_input2)
z21 = boundary_model_tmp.layers[4](z21_input)
z22 = boundary_model_tmp.layers[4](z22_input)
x2 = boundary_model_tmp.layers[5]([c2,z21,z22])


################################Prediction of boundaries and directionality ###############

direction_model = load_model(boundary_direction_file)
direction_model.name = 'boundary_direction_model'



dir_model_layers = [k for k in direction_model.layers]

'''Run it through fully-connected layers of the boundary orientation model to get left/right boundary prediction'''
d1, d2 = x1, x2
for i in range(len(dir_model_layers) - 10, len(dir_model_layers)):
    dir_model_layers[i].trainable = False
    dir_model_layers[i].name = 'd' + str(i)
    d1 = dir_model_layers[i](d1)
    d2 = dir_model_layers[i](d2)
    

d = Subtract()([d1, d2])


boundary_model = load_model(boundary_model_file)
boundary_model.name = 'boundary_model'

boundary_model_layers = [k for k in boundary_model.layers]
'''Run it through fully-connected layers of the boundary model to get prediction of boundary'''
b1, b2 = x1, x2
for i in range(len(boundary_model_layers) - 10, len(boundary_model_layers)):
    boundary_model_layers[i].trainable = False
    boundary_model_layers[i].name = 'b' + str(i)
    b1 = boundary_model_layers[i](b1)
    b2 = boundary_model_layers[i](b2)

b = Subtract()([x1, x2])

######################################## append fully-connected layers for loop model
x = concatenate([b, x1, x2])

x = Dense(units = 512)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)

x = Dense(units = 256)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)

pred = Dense(units = 1, activation='sigmoid')(x)


if rnn_len > 0:
    model = Model(inputs = [seq_input1, z11_input, z12_input, seq_input2, z21_input, z22_input], outputs = pred)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])
    
parallel_model = multi_gpu_model(model, gpus=6)
#parallel_model = model

parallel_model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy', common_function_evaluation.fbeta_score])

parallel_model.summary()


if existing_model_file:
    parallel_model.load_weights(existing_model_file)

################## Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

roc = roc_callback.ROC_Callback(val_generator = val_generator, fout=flog)
callback_list = [checkpoint, roc, early_stopping]


parallel_model.fit_generator(generator = train_generator, epochs= epoch, shuffle=True, 
          validation_data = val_generator, use_multiprocessing=False, workers=1,
          callbacks = callback_list, verbose=1)



################## Testing
parallel_model.load_weights(output_best_model)

parallel_model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])



print('best model monitored')
flog.write('best model monitored')

# parallel_model.save(output_model_cnn)
# parallel_model.save_weights(output_best_model)

model.save(output_model_cnn.replace('.h5','_1gpu.h5'))
model.save_weights(output_best_model.replace('.h5','_1gpu.h5'))

common_function_evaluation.evaluate(parallel_model, test_generator, title = 'Loop prediction',label='GM12878 test', curvefile='gm12878_loop_test_nopooling')


#########
flog.close()



















