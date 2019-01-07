#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:12:37 2018


Boundary prediction from pre-trained CNN + pre-trained LSTM
@author: tat2016
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout, BatchNormalization,\
                        Input, concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU

from keras.utils import multi_gpu_model

import data_generator
import roc_callback

from datetime import datetime

import common_function_evaluation


#####################
epoch = 15

batch_size = 64



log_file = "log/boundary_prediction_log_%s.txt" % (datetime.now().strftime('%H_%M_%d_%m_%Y'))

cna = 2 #copy number alteration

leaky = 0.2
dropout = 0.2

rnn_len = 800 #sequence length to run RNN, set to 0 to disable using RNN

segment_size = 4000 #length of boundaries

version = ''

cnn_model_file = 'boundary_pred_4k_1gpu.h5' #'boundary_pred_4k_1gpu.h5'
lstm_model_file = 'boundary_pred_4k_lstm800_1gpu.h5' #'boundary_pred_4k_2lstm800_1gpu.h5'


existing_model = ''

output_model_cnn = "boundary_pred_%dk%s_cnnlstm_hybrid%d.h5" % (segment_size/1000, version, rnn_len)
output_best_model = "boundary_pred_%dk_best%s_cnnlstm_hybrid%d.h5" % (segment_size/1000, version, rnn_len)


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




def loadPreTrainedModel(model_file):
    model = load_model(model_file)
    for _ in range(10):
        model.layers.pop()
    
    model.outputs = [model.layers[-1].output]
    
    model = Model(model.inputs, model.outputs)
    model.trainable = False
    return(model)

    

cnn_model = loadPreTrainedModel(cnn_model_file)
lstm_model = loadPreTrainedModel(lstm_model_file)



seq_input = Input(shape=(segment_size,nbr_feature,1)) 
z1_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn1') #copy number1
z2_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn2') #copy number2
  
z1 = lstm_model(z1_input)
z2 = lstm_model(z2_input)

x = cnn_model(seq_input)

x = concatenate([x, z1, z2])


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

inputs = [seq_input, z1_input, z2_input]
    

model = Model(inputs= inputs , outputs=predictions)


pmodel = multi_gpu_model(model, gpus=4)
#pmodel = model

pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy', common_function_evaluation.fbeta_score])

pmodel.summary()


model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])

if existing_model:
    pmodel.load_weights(existing_model)


############## Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

roc = roc_callback.ROC_Callback(val_generator = val_generator, fout=flog)
callback_list = [checkpoint, roc, early_stopping]

trained_history = pmodel.fit_generator(generator = train_generator, epochs= epoch, shuffle=True, 
          validation_data = val_generator, use_multiprocessing=False, workers=1,
          callbacks = callback_list, verbose=1)


pmodel.load_weights(output_best_model)
pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])



#############Testing
print('best model monitored')
flog.write('best model monitored')

common_function_evaluation.evaluate(pmodel, test_generator, title = 'Boundary prediction',label='GM12878 test', curvefile='cnnlstm_gm12878_test')



pmodel.save(output_model_cnn)
pmodel.save_weights(output_best_model)

model.save(output_model_cnn.replace('.h5','_1gpu.h5'))
model.save_weights(output_best_model.replace('.h5','_1gpu.h5'))


#########
flog.close()

version = 'negChIP'
output_file_test_data = 'data_boundary_%dk_test%s.mat' % (segment_size/1000, version)
output_file_test_label = 'label_boundary_%dk_test%s.mat' % (segment_size/1000, version)
test_generator = data_generator.DataGenerator(output_file_test_data, output_file_test_label, shuffle = False, use_reverse = True, **params) #set shuffle=False to calculate AUC
common_function_evaluation.evaluate(pmodel, test_generator, title = 'Boundary prediction',label='GM12878 test (nopooling)')

