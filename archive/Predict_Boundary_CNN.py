#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:16:54 2018

@author: tat2016

CNN model to predict boundary from sequence only
"""


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


import matplotlib.pyplot as plt
plt.switch_backend('agg')


from keras.models import Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization,\
                        Flatten,Input,Conv2D, MaxPooling2D

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU


from keras.utils import multi_gpu_model

import data_generator
import roc_callback

from datetime import datetime
import common_function_evaluation


##############################    
epoch = 15
batch_size = 48



segment_size = 4000 #length of boundaries


log_file = "log/boundary_prediction_log_%s.txt" % (datetime.now().strftime('%H_%M_%d_%m_%Y'))

cna = 2 #copy number alteration

leaky = 0.2
dropout = 0.2

rnn_len = 0 #sequence length in input data, set to 0 to disable using RNN


version = '' #to specify training data

existing_model = ''

output_model_cnn = "boundary_pred_%dk%s.h5" % (segment_size/1000, version)
output_best_model = "boundary_pred_%dk_best%s.h5" % (segment_size/1000, version)


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


########## Loading data
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len}

train_generator = data_generator.DataGenerator(file_train_data, file_train_label, shuffle = True, **params)
val_generator = data_generator.DataGenerator(file_val_data, file_val_label, shuffle = False, **params) #set shuffle=False to calculate AUC
test_generator = data_generator.DataGenerator(file_test_data, file_test_label, shuffle = False, use_reverse = False, **params) #set shuffle=False to calculate AUC



########## Build network
seq_input = Input(shape=(segment_size,nbr_feature,1)) 


initializer = 'glorot_uniform'

x = Conv2D(filters=1024, kernel_size = (17,5), strides = (1,5), input_shape=(segment_size, nbr_feature,1), 
           padding='valid', kernel_initializer = initializer)(seq_input)

x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout) (x)


x = Conv2D(filters=1024, kernel_size = (5,1), strides=(1,1), input_shape=(segment_size, cna, 1), 
           padding='same', kernel_initializer = initializer)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = MaxPooling2D(pool_size=(segment_size - 16,1), strides=(segment_size - 16,1), padding='valid')(x)
x = Dropout(dropout) (x)

x = Flatten()(x)

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


inputs = [seq_input]


model = Model(inputs= inputs , outputs=predictions)


pmodel = multi_gpu_model(model, gpus=2)
#pmodel = model

pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy', common_function_evaluation.fbeta_score])

pmodel.summary()


model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])

if existing_model:
    pmodel.load_weights(existing_model)

################ Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

roc = roc_callback.ROC_Callback(val_generator = val_generator, fout=flog)
callback_list = [checkpoint, roc, early_stopping]

trained_history = pmodel.fit_generator(generator = train_generator, epochs= epoch, shuffle=True, 
          validation_data = val_generator, use_multiprocessing=False, workers=1,
          callbacks = callback_list, verbose=1)



################ Test with best trained model
pmodel.load_weights(output_best_model)
# Compile model (required to make predictions)
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




















