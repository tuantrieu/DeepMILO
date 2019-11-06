#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:16:54 2018

@author: tat2016

'boundary orientation model' to predict boundary directionality from sequence only
"""

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import multi_gpu_model

from keras.models import load_model
from keras.layers import Input

import keras.backend as K



import data_generator
import roc_callback

from constant import *


from datetime import datetime
import evaluation_function as ef

segment_size = REGION_SIZE
epoch = 15
ngpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

batch_size = 54

leaky = 0.2
dropout = 0.1

rnn_len = 800  # sequence length to run RNN, set to 0 to disable using RNN

version = ''


boundary_model_file = 'model/boundary_4k_sepcnnlstm_ruan_1gpu.h5'

existing_model = ''

output_model = 'model/direction_pred_sep_4k_ruan.h5'
output_best_model = 'model/direction_pred_sep_4k_best_ruan.h5'

file_train_data = 'data/data_boundary_direction_4k_train_ruan.mat'
file_train_label = 'data/label_boundary_direction_4k_train_ruan.mat'

file_val_data = 'data/data_boundary_direction_4k_val_ruan.mat'
file_val_label = 'data/label_boundary_direction_4k_val_ruan.mat'

file_test_data = 'data/data_boundary_direction_4k_test_ruan.mat'
file_test_label = 'data/label_boundary_direction_4k_test_ruan.mat'


nbr_feature = len(LETTERTOINDEX)# number of features


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

train_generator = data_generator.DataGenerator(file_train_data, file_train_label, shuffle=True, use_reverse=False,
                                               **params)
val_generator = data_generator.DataGenerator(file_val_data, file_val_label, shuffle=False, use_reverse=False,
                                             **params)  # set shuffle=False to calculate AUC
test_generator = data_generator.DataGenerator(file_test_data, file_test_label, shuffle=False, use_reverse=False,
                                              **params)  # set shuffle=False to calculate AUC

###########Best architecture

seq_input = Input(shape=(segment_size, nbr_feature, 1))
z1_input = Input(shape=(rnn_len, nbr_feature), name='rnn1')


boundary_model = load_model(boundary_model_file)

for layer in boundary_model.layers[:10]:
    layer.trainable = False

#reset weights of fully-connected layers to re-train
session = K.get_session()
for layer in boundary_model.layers[10:]:
    if hasattr(layer, 'kernel'):
        layer.kernel.initializer.run(session=session)


model = boundary_model
model.summary()

pmodel = multi_gpu_model(model, gpus=ngpu)

pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy', ef.fbeta_score])


if existing_model:
    pmodel.load_weights(existing_model)

############### Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='min')

roc = roc_callback.ROC_Callback(val_generator=val_generator, fout=flog)
callback_list = [checkpoint, roc, early_stopping]

trained_history = pmodel.fit_generator(generator=train_generator, epochs=epoch, shuffle=True,
                                       validation_data=val_generator, use_multiprocessing=False, workers=1,
                                       callbacks=callback_list, verbose=1)

############## Testing
pmodel.load_weights(output_best_model)
pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
               metrics=['accuracy', ef.fbeta_score])

print('best model monitored')
flog.write('best model monitored')

ef.evaluate(pmodel, test_generator, flog=flog)

pmodel.save(output_model)
pmodel.save_weights(output_best_model)

model.save(output_model.replace('.h5', '_1gpu.h5'))
model.save_weights(output_best_model.replace('.h5', '_1gpu.h5'))

#########
flog.close()
















