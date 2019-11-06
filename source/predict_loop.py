#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:16:54 2018

@author: tat2016

Loop model to predict loop from sequence only

Note: when training on multiple GPU, loading into a single GPU results in error.
Try to save the single-GPU model (before it is copied to multiple GPUs), it shares the same weights
"""

import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

import keras
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, BatchNormalization, \
    Input, concatenate, Subtract

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU

from keras.utils import multi_gpu_model

import data_generator
import roc_callback

from datetime import datetime


from constant import *
import evaluation_function as ef

################ Input

segment_size = REGION_SIZE
epoch = 5

ngpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

batch_size = 12 * ngpu

leaky = 0.2
dropout = 0.00

rnn_len = 800  # sequence length to run RNN, set to 0 to disable using RNN

version = ''

existing_model = '' #'model/loop_pred_4k_ruan_sep_1gpu.h5'  # 'loop_pred_4k_best_weight.h5'

boundary_model_file = 'model/boundary_4k_sepcnnlstm_ruan_1gpu.h5'  # 'boundary_4k_cnnlstm_ruan_1gpu.h5.h5'
boundary_direction_file = 'model/direction_pred_sep_4k_ruan_1gpu.h5'  # 'direction_pred_4k_ruan.h5'

output_model_cnn = "model/loop_pred_4k_ruan_sep_unfreeze.h5"
output_best_model = "model/loop_pred_4k_best_ruan_sep_unfreeze.h5"

file_train_data = 'data/data_loop_4k_train_ruan.mat'
file_train_label = 'data/label_loop_4k_train_ruan.mat'

# file_train_data = 'data/data_loop_4k_val_ruan.mat'
# file_train_label = 'data/label_loop_4k_val_ruan.mat'

file_val_data = 'data/data_loop_4k_val_ruan.mat'
file_val_label = 'data/label_loop_4k_val_ruan.mat'

file_test_data = 'data/data_loop_4k_test_ruan.mat'
file_test_label = 'data/label_loop_4k_test_ruan.mat'


nbr_feature = len(LETTERTOINDEX)  # number of features

script_name = sys.argv[0].replace('.py','').split('/')[-1]

log_file = "log/%s_log_%s.txt" % (script_name, datetime.now().strftime('%H_%M_%d_%m_%Y'))

flog = open(log_file,"w")

flog.write('existing model:{}\n'.format(existing_model))
flog.write('leaky:{}, dropout:{}, rnnlen: {}, segment_size:{}\n'.format(leaky, dropout, rnn_len, segment_size))
flog.write('version:{}, existing model:{}\n'.format(version, existing_model))
flog.write('train data: {}, {}\n'.format(file_train_data, file_train_label))
flog.flush()


##########
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len}

train_generator = data_generator.DataGeneratorLoopSeq(file_train_data, file_train_label, shuffle=True, **params)
val_generator = data_generator.DataGeneratorLoopSeq(file_val_data, file_val_label, shuffle=False,
                                                    **params)  # set shuffle=False to calculate AUC
test_generator = data_generator.DataGeneratorLoopSeq(file_test_data, file_test_label, shuffle=False, use_reverse=False,
                                                     **params)  # set shuffle=False to calculate AUC

################################################
'''Boundary 1'''
seq_input1 = Input(shape=(segment_size, nbr_feature, 1))
z1_input = Input(shape=(rnn_len, nbr_feature), name='rnn1')


'''Boundary 2'''
seq_input2 = Input(shape=(segment_size, nbr_feature, 1))
z2_input = Input(shape=(rnn_len, nbr_feature), name='rnn2')

################################################
boundary_model_tmp = load_model(boundary_model_file)

boundary_model_tmp.layers[4].name = 'boundary_cat'

# try unfreeze it after retrain
# boundary_model_tmp.layers[2].trainable = False
# boundary_model_tmp.layers[3].trainable = False
# boundary_model_tmp.layers[4].trainable = False

'''running through CNN and RNN(LSTM) in the boundary model'''
c1 = boundary_model_tmp.layers[2](seq_input1)
z1 = boundary_model_tmp.layers[3](z1_input)
x1 = boundary_model_tmp.layers[4]([c1, z1])

c2 = boundary_model_tmp.layers[2](seq_input2)
z2 = boundary_model_tmp.layers[3](z2_input)
x2 = boundary_model_tmp.layers[4]([c2, z2])

################################Prediction of boundaries and directionality ###############

direction_model = load_model(boundary_direction_file)
direction_model.name = 'boundary_direction_model'

dir_model_layers = [k for k in direction_model.layers]

'''Run it through fully-connected layers of the boundary orientation model to get left/right boundary prediction'''
d1, d2 = x1, x2
for i in range(len(dir_model_layers) - 10, len(dir_model_layers)):
    # try unfreeze it after retrain
    # dir_model_layers[i].trainable = False
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
    # try unfreeze it after retrain
    # boundary_model_layers[i].trainable = False
    boundary_model_layers[i].name = 'b' + str(i)
    b1 = boundary_model_layers[i](b1)
    b2 = boundary_model_layers[i](b2)

b = Subtract()([b1, b2])

######################################## append fully-connected layers for loop model
x = concatenate([b, d, x1, x2])

x = Dense(units=512)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)

x = Dense(units=256)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)

pred = Dense(units=1, activation='sigmoid')(x)

if rnn_len > 0:
    model = Model(inputs=[seq_input1, z1_input, seq_input2, z2_input], outputs=pred)


if existing_model:
    model.load_weights(existing_model)

parallel_model = multi_gpu_model(model, gpus=ngpu)

parallel_model.summary()


optimizer = keras.optimizers.RMSprop(lr=1e-4)
parallel_model.compile(loss='binary_crossentropy', optimizer=optimizer,
                       metrics=['accuracy', ef.fbeta_score])


################## Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

roc = roc_callback.ROC_Callback(val_generator=val_generator, fout=flog)
callback_list = [checkpoint, roc, early_stopping]

parallel_model.fit_generator(generator=train_generator, epochs=epoch, shuffle=True,
                             validation_data=val_generator, use_multiprocessing=False, workers=1,
                             callbacks=callback_list, verbose=1)

################## Testing
parallel_model.load_weights(output_best_model)

parallel_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                       metrics=['accuracy', ef.fbeta_score])

print('best model monitored')
flog.write('best model monitored')

parallel_model.save(output_model_cnn)
parallel_model.save_weights(output_best_model)

model.save(output_model_cnn.replace('.h5', '_1gpu.h5'))
model.save_weights(output_best_model.replace('.h5', '_1gpu.h5'))

ef.evaluate(parallel_model, test_generator)

#########
flog.close()





















