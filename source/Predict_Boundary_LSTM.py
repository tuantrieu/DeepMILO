#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:16:54 2018

@author: tat2016

LSTM model to predict boundary from sequence only
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import matplotlib.pyplot as plt
plt.switch_backend('agg')


from keras.models import Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization,\
                        Flatten,Input, Bidirectional, LSTM, TimeDistributed

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU

from keras.utils import multi_gpu_model

import data_generator
import roc_callback

from datetime import datetime
import common_function_evaluation



######################
segment_size = 4000


epoch = 0

batch_size = 52


log_file = "log/boundary_prediction_log_%s.txt" % (datetime.now().strftime('%H_%M_%d_%m_%Y'))

cna = 2 #copy number alteration

leaky = 0.2
dropout = 0.1

rnn_len = 800 #sequence length to run RNN, set to 0 to disable using RNN


version = ''

existing_model = ''

output_model_cnn = "boundary_pred_%dk%s_lstm%d.h5" % (segment_size/1000, version, rnn_len)
output_best_model = "boundary_pred_%dk_best%s_lstm%d.h5" % (segment_size/1000, version, rnn_len)


file_train_data = 'data_boundary_%dk%s_train.mat' % (segment_size/1000, version)
file_train_label = 'label_boundary_%dk%s_train.mat' % (segment_size/1000, version)


file_val_data = 'data_boundary_%dk%s_val.mat' % (segment_size/1000, version)
file_val_label = 'label_boundary_%dk%s_val.mat' % (segment_size/1000, version)

file_test_data = 'data_boundary_%dk%s_test.mat' % (segment_size/1000, version)
file_test_label = 'label_boundary_%dk%s_test.mat' % (segment_size/1000,version)



all_letters = 'ACGTN'
n_letters = len(all_letters)

nbr_feature = n_letters * 2# number of features

flog = open(log_file,"w")


##########
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len}

train_generator = data_generator.DataGenerator(file_train_data, file_train_label, shuffle = True, **params)
val_generator = data_generator.DataGenerator(file_val_data, file_val_label, shuffle = False, **params) #set shuffle=False to calculate AUC
test_generator = data_generator.DataGenerator(file_test_data, file_test_label, shuffle = False, use_reverse = False, **params) #set shuffle=False to calculate AUC





    
########### architecture

seq_input = Input(shape=(segment_size,nbr_feature,1)) 


initializer = 'glorot_uniform'


if rnn_len > 0:
    
    z1_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn1')

    blstm1 = Bidirectional(LSTM(128, return_sequences=True,dropout=dropout))
    blstm2 = Bidirectional(LSTM(128, return_sequences=True,dropout=dropout))
    timeDist = TimeDistributed(Dense(1, activation='sigmoid'))
    
    z1 = blstm1(z1_input)
    z1 = blstm2(z1)
    z1 = timeDist(z1)
    z1 = Flatten()(z1)

    z2_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn2') #unused to save training time
#    z2 = blstm1(z2_input)
#    z2 = blstm2(z2)
#    z2 = timeDist(z2)
#    z2 = Flatten()(z2)



if rnn_len > 0:

    #x = concatenate([z1, z2])
    x = z1

x = Dense(units = 512)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)


x = Dense(units = 256)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)


x = Dense(units = 1)(x)

predictions = Activation('sigmoid')(x)

if rnn_len > 0:
    inputs = [seq_input, z1_input, z2_input]
    

model = Model(inputs= inputs , outputs=predictions)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])


pmodel = multi_gpu_model(model, gpus=4)
#pmodel = model

pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy', common_function_evaluation.fbeta_score])

pmodel.summary()


if existing_model:
    pmodel.load_weights(existing_model)


################# Training
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

roc = roc_callback.ROC_Callback(val_generator = val_generator, fout=flog)
callback_list = [checkpoint, roc, early_stopping]

trained_history = pmodel.fit_generator(generator = train_generator, epochs= epoch, shuffle=True, 
          validation_data = val_generator, use_multiprocessing=False, workers=3,
          callbacks = callback_list, verbose=1)




################ Testing
pmodel.load_weights(output_best_model)
pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])


print('best model monitored')
flog.write('best model monitored')

common_function_evaluation.evaluate(pmodel, test_generator, title = 'Boundary prediction',label='GM12878 test (nopooling)', curvefile='gm12878_test')


# pmodel.save(output_model_cnn)
# pmodel.save_weights(output_best_model)

model.save(output_model_cnn.replace('.h5','_1gpu.h5'))
model.save_weights(output_best_model.replace('.h5','_1gpu.h5'))


#########
flog.close()



















        
