#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:12:37 2018


Boundary prediction from pre-trained CNN + pre-trained LSTM
@author: tat2016
"""

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, \
    Input, concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU

from keras.utils import multi_gpu_model

import data_generator
import roc_callback

from datetime import datetime

import evaluation_function as ef


'''------------Parameters--------------'''

version = ''

ngpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
epoch = 15

batch_size = 64

leaky = 0.2
dropout = 0.2

rnn_len = 800  # sequence length to run RNN, set to 0 to disable using RNN

segment_size = 4000  # length of boundaries


cnn_model_file = 'model/boundary_sep_cnn_4k_loocv_test_1gpu.h5'
lstm_model_file = 'model/boundary_lstm_4k_800_loocv_1gpu.h5'

output_model = "model/boundary_4k_sepcnnlstm_loocv_gm12878.h5"
output_best_model = "model/boundary_4k_sepcnnlstm_best_loocv_gm12878.h5"

cohort = 'mcf7'
suffix = '_loocv'

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

existing_model = ''


# output_model = "model/boundary_4k_sepcnnlstm_ruan.h5"
# output_best_model = "model/boundary_4k_sepcnnlstm_best_ruan.h5"
# cnn_model_file = 'model/boundary_sep_cnn_4k_ruan_test_1gpu.h5'
# lstm_model_file = 'model/boundary_lstm_4k_800_ruan_1gpu.h5'
#
# print('cnn model:{}, lstm model:{}'.format(cnn_model_file, lstm_model_file))
#
# existing_model = ''
#
# file_train_data = 'data/data_boundary_4k_train_ruan.mat'
# file_train_label = 'data/label_boundary_4k_train_ruan.mat'
#
# file_val_data = 'data/data_boundary_4k_val_ruan.mat'
# file_val_label = 'data/label_boundary_4k_val_ruan.mat'
#
# file_test_data = 'data/data_boundary_4k_test_ruan.mat'
# file_test_label = 'data/label_boundary_4k_test_ruan.mat'


nbr_feature = 5  # number of features

script_name = sys.argv[0].replace('.py','')

log_file = "log/%s_log_%s.txt" % (script_name, datetime.now().strftime('%H_%M_%d_%m_%Y'))

flog = open(log_file, "w")

flog.write('cnn model:{}, lstm model:{}'.format(cnn_model_file, lstm_model_file))
flog.write('leaky:{}, dropout:{}, rnnlen: {}, segment_size:{}\n'.format(leaky, dropout, rnn_len, segment_size))
flog.write('version:{}, existing model:{}\n'.format(version, existing_model))
flog.write('train data: {}, {}\n'.format(file_train_data, file_train_label))
flog.flush()
##########
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len}

train_generator = data_generator.DataGenerator(file_train_data, file_train_label, shuffle=True, **params)
val_generator = data_generator.DataGenerator(file_val_data, file_val_label, shuffle=False,
                                             **params)  # set shuffle=False to calculate AUC
# test_generator = data_generator.DataGenerator(file_test_data, file_test_label, shuffle=False, use_reverse=False,
#                                               **params)  # set shuffle=False to calculate AUC

test_generator1 = data_generator.DataGenerator(file_test_data1, file_test_label1, shuffle=False, use_reverse=False, **params) #set shuffle=False to calculate AUC
test_generator2 = data_generator.DataGenerator(file_test_data2, file_test_label2, shuffle=False, use_reverse=False, **params) #set shuffle=False to calculate AUC
test_generator3 = data_generator.DataGenerator(file_test_data3, file_test_label3, shuffle=False, use_reverse=False, **params) #set shuffle=False to calculate AUC


def load_pretrained_model(model_file):
    pre_model = load_model(model_file)
    for _ in range(10):
        pre_model.layers.pop()

    pre_model.outputs = [pre_model.layers[-1].output]

    pre_model = Model(pre_model.inputs, pre_model.outputs)
    pre_model.trainable = False
    return pre_model


cnn_model = load_pretrained_model(cnn_model_file)
lstm_model = load_pretrained_model(lstm_model_file)

seq_input = Input(shape=(segment_size, nbr_feature, 1))
z1_input = Input(shape=(rnn_len, nbr_feature), name='rnn1')  # copy number1

z1 = lstm_model(z1_input)

x = cnn_model(seq_input)

x = concatenate([x, z1])

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

inputs = [seq_input, z1_input]

model = Model(inputs=inputs, outputs=predictions)

pmodel = multi_gpu_model(model, gpus=ngpu)
# pmodel = model

pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
               metrics=['accuracy', ef.fbeta_score])

pmodel.summary()


if existing_model:
    pmodel.load_weights(existing_model)

############## Training
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='min')

roc = roc_callback.ROC_Callback(val_generator=val_generator, fout=flog)
callback_list = [checkpoint, roc, early_stopping]

trained_history = pmodel.fit_generator(generator=train_generator, epochs=epoch, shuffle=True,
                                       validation_data=val_generator, use_multiprocessing=False, workers=1,
                                       callbacks=callback_list, verbose=1)

pmodel.load_weights(output_best_model)
pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', ef.fbeta_score])

#############Testing
print('best model monitored')
flog.write('best model monitored')


# ef.evaluate(pmodel, test_generator, flog=flog)

ef.evaluate(pmodel, test_generator1, flog=flog, name=file_test_label1, output_file=file_test_label1.replace('.mat','_sepcnnlstm_output.txt'))
ef.evaluate(pmodel, test_generator2, flog=flog, name=file_test_label2, output_file=file_test_label2.replace('.mat','_sepcnnlstm_output.txt'))
ef.evaluate(pmodel, test_generator3, flog=flog, name=file_test_label3, output_file=file_test_label3.replace('.mat','_sepcnnlstm_output.txt'))


pmodel.save(output_model)
pmodel.save_weights(output_best_model)

model.save(output_model.replace('.h5', '_1gpu.h5'))
model.save_weights(output_best_model.replace('.h5', '_1gpu.h5'))

#########
flog.close()

# version = 'negChIP'
# output_file_test_data = 'data_boundary_%dk_test%s.mat' % (segment_size / 1000, version)
# output_file_test_label = 'label_boundary_%dk_test%s.mat' % (segment_size / 1000, version)
# test_generator = data_generator.DataGenerator(output_file_test_data, output_file_test_label, shuffle=False,
#                                               use_reverse=True, **params)  # set shuffle=False to calculate AUC
# ef.evaluate(pmodel, test_generator, title='Boundary prediction',
#                                     label='GM12878 test (nopooling)')

