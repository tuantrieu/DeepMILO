#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:16:54 2018

@author: tat2016

LSTM model to predict boundary from sequence only
"""

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


import matplotlib.pyplot as plt


from keras.models import Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, \
    Flatten, Input, Bidirectional, LSTM, TimeDistributed

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU

from keras.utils import multi_gpu_model

import data_generator
import roc_callback

from datetime import datetime
import evaluation_function as ef

plt.switch_backend('agg')

######################
segment_size = 4000
epoch = 20
batch_size = 64
nbr_feature = 5

existing_model = 'model/boundary_lstm_4k_best_800_all.h5'

ngpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

log_file = "log/boundary_prediction_log_%s.txt" % (datetime.now().strftime('%H_%M_%d_%m_%Y'))

use_mutation = True if len(sys.argv) > 1 and sys.argv[1] == 'mutation' else False

version = ''

leaky = 0.2
dropout = 0.2

rnn_len = 800  # sequence length to run RNN, set to 0 to disable using RNN

cohort = 'k562'
suffix = '_loocv'


# output_model_cnn = "model/boundary_lstm_4k_800_ruan.h5"
# output_best_model = "model/boundary_lstm_4k_800_best_ruan.h5"
#
# file_train_data = 'data/data_boundary_4k_train_ruan.mat'
# file_train_label = 'data/label_boundary_4k_train_ruan.mat'
#
# file_val_data = 'data/data_boundary_4k_val_ruan.mat'
# file_val_label = 'data/label_boundary_4k_val_ruan.mat'
#
# file_test_data = 'data/data_boundary_4k_test_ruan.mat'
# file_test_label = 'data/label_boundary_4k_test_ruan.mat'

output_model_cnn = "model/boundary_lstm_4k_800_{}.h5".format(suffix + cohort)
output_best_model = "model/boundary_lstm_4k_800_best_{}.h5".format(suffix + cohort)

file_train_data = 'data/data_boundary_4k_train_leftout_{}.mat'.format(cohort)
file_train_label = 'data/label_boundary_4k_train_leftout_{}.mat'.format(cohort)

file_val_data = 'data/data_boundary_4k_val_leftout_{}.mat'.format(cohort)
file_val_label = 'data/label_boundary_4k_val_leftout_{}.mat'.format(cohort)

file_test_data1 = 'data/data_boundary_4k_test_type1_leftout_{}.mat'.format(cohort)
file_test_label1 = 'data/label_boundary_4k_test_type1_leftout_{}.mat'.format(cohort)
file_test_data2 = 'data/data_boundary_4k_test_type2_leftout_{}.mat'.format(cohort)
file_test_label2 = 'data/label_boundary_4k_test_type2_leftout_{}.mat'.format(cohort)
file_test_data3 = 'data/data_boundary_4k_test_type3_leftout_{}.mat'.format(cohort)
file_test_label3 = 'data/label_boundary_4k_test_type3_leftout_{}.mat'.format(cohort)




nbr_feature = 5  # len('ACGTN') number of features

script_name = sys.argv[0].replace('.py','')
log_file = "log/%s_log_%s.txt" % (script_name, datetime.now().strftime('%H_%M_%d_%m_%Y'))

flog = open(log_file, "w")
flog.write('leaky:{}, dropout:{}, rnnlen: {}, segment_size:{}\n'.format(leaky, dropout, rnn_len, segment_size))
flog.write('version:{}, existing model:{}\n'.format(version, existing_model))
flog.write('train data: {}, {}\n'.format(file_train_data, file_train_label))

##########
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len,
          'rnn_only': True}


train_generator = data_generator.DataGenerator(file_train_data, file_train_label, shuffle=True, **params)
val_generator = data_generator.DataGenerator(file_val_data, file_val_label, shuffle=False,
                                             **params)  # set shuffle=False to calculate AUC
# test_generator = data_generator.DataGenerator(file_test_data, file_test_label, shuffle=False, use_reverse=False,
#                                               **params)  # set shuffle=False to calculate AUC

test_generator1 = data_generator.DataGenerator(file_test_data1, file_test_label1, shuffle=False, use_reverse=False,
                                              **params)  # set shuffle=False to calculate AUC
test_generator2 = data_generator.DataGenerator(file_test_data2, file_test_label2, shuffle=False, use_reverse=False,
                                              **params)  # set shuffle=False to calculate AUC
test_generator3 = data_generator.DataGenerator(file_test_data3, file_test_label3, shuffle=False, use_reverse=False,
                                              **params)  # set shuffle=False to calculate AUC

########### architecture

initializer = 'glorot_uniform'

z1_input = Input(shape=(rnn_len, nbr_feature), name='rnn1')

blstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout))
blstm2 = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout))
timeDist = TimeDistributed(Dense(1, activation='sigmoid'))

z1 = blstm1(z1_input)
z1 = blstm2(z1)
z1 = timeDist(z1)
z1 = Flatten()(z1)

x = z1

x = Dense(units=512)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)

x = Dense(units=256)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)

x = Dense(units=1)(x)

predictions = Activation('sigmoid')(x)


model = Model(inputs=z1_input, outputs=predictions)

pmodel = multi_gpu_model(model, gpus=ngpu)
# pmodel = model

pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
               metrics=['accuracy', ef.fbeta_score])

pmodel.summary()

if existing_model:
    pmodel.load_weights(existing_model)

################# Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='min')

roc = roc_callback.ROC_Callback(val_generator=val_generator, fout=flog)
callback_list = [checkpoint, roc, early_stopping]


pmodel.fit_generator(generator=train_generator, epochs=epoch, shuffle=True,
                                       validation_data=val_generator, use_multiprocessing=False, workers=ngpu,
                                       callbacks=callback_list, verbose=1)

################ Testing
pmodel.load_weights(output_best_model)
pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
               metrics=['accuracy', ef.fbeta_score])

print('best model monitored')
flog.write('best model monitored')

pmodel.save(output_model_cnn)
pmodel.save_weights(output_best_model)

model.save(output_model_cnn.replace('.h5', '_1gpu.h5'))
model.save_weights(output_best_model.replace('.h5', '_1gpu.h5'))

#ef.evaluate(pmodel, test_generator, flog=flog)

ef.evaluate(pmodel, test_generator1, flog=flog, name=file_test_label1, output_file=file_test_label1.replace('.mat','_lstm_output.txt'))
ef.evaluate(pmodel, test_generator2, flog=flog, name=file_test_label2, output_file=file_test_label2.replace('.mat','_lstm_output.txt'))
ef.evaluate(pmodel, test_generator3, flog=flog, name=file_test_label3, output_file=file_test_label3.replace('.mat','_lstm_output.txt'))


#########
flog.close()




















