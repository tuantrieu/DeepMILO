#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:16:54 2018

@author: tat2016

'boundary orientation model' to predict boundary directionality from sequence only
"""



import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"



import matplotlib.pyplot as plt
plt.switch_backend('agg')


from keras.models import load_model
from keras.layers import Input

from keras.callbacks import EarlyStopping, ModelCheckpoint


from keras.utils import multi_gpu_model

from keras import backend as K

import data_generator
import roc_callback

import common_function_evaluation

from datetime import datetime



segment_size = 4000
epoch = 10

batch_size = 54


log_file = "log/direction_prediction_log_%s.txt" % (datetime.now().strftime('%H_%M_%d_%m_%Y'))

cna = 2 #copy number alteration

leaky = 0.2
dropout = 0.1

rnn_len = 800 #sequence length to run RNN, set to 0 to disable using RNN


version = ''


boundary_model_file = 'boundary_pred_4k_cnnlstm800_1gpu.h5'

existing_model = ''

output_model_cnn = 'direction_pred_%dk%s.h5' % (segment_size/1000, version)
output_best_model = 'direction_pred_%dk_best%s.h5' % (segment_size/1000, version)


file_train_data = 'data_boundary_direction_%dk_train%s.mat' % (segment_size/1000, version)
file_train_label = 'label_boundary_direction_%dk_train%s.mat' % (segment_size/1000, version)

file_val_data = 'data_boundary_direction_%dk_val%s.mat' % (segment_size/1000, version)
file_val_label = 'label_boundary_direction_%dk_val%s.mat' % (segment_size/1000, version)

file_test_data = 'data_boundary_direction_%dk_test%s.mat' % (segment_size/1000, version)
file_test_label = 'label_boundary_direction_%dk_test%s.mat' % (segment_size/1000,version)



all_letters = 'ACGTN'
n_letters = len(all_letters)

nbr_feature = n_letters * 2# number of features

flog = open(log_file,"w")


##########
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len}

train_generator = data_generator.DataGenerator(file_train_data, file_train_label, shuffle = True, use_reverse = False, **params)
val_generator = data_generator.DataGenerator(file_val_data, file_val_label, shuffle = False, use_reverse = False, **params) #set shuffle=False to calculate AUC
test_generator = data_generator.DataGenerator(file_test_data, file_test_label, shuffle = False, use_reverse = False, **params) #set shuffle=False to calculate AUC


###########Best architecture
  
seq_input = Input(shape=(segment_size,nbr_feature,1)) 
z1_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn1')
z2_input = Input(shape=(rnn_len, int(nbr_feature/2)), name='rnn2')


boundary_model = load_model(boundary_model_file)


for layer in boundary_model.layers[:10]:
    layer.trainable = False


#reset weights of fully-connected layers to re-train
session = K.get_session()
for layer in boundary_model.layers[10:]:
    if hasattr(layer, 'kernel'):
        layer.kernel.initializer.run(session=session)
    
    
model = boundary_model

pmodel = multi_gpu_model(model, gpus=6)
#pmodel = boundary_model

pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy', common_function_evaluation.fbeta_score])

pmodel.summary()


model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])


if existing_model:
    pmodel.load_weights(existing_model)


############### Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

roc = roc_callback.ROC_Callback(val_generator = val_generator, fout=flog)
callback_list = [checkpoint, roc, early_stopping]

trained_history = pmodel.fit_generator(generator = train_generator, epochs= epoch, shuffle=True, 
          validation_data = val_generator, use_multiprocessing=False, workers=1,
          callbacks = callback_list, verbose=1)


############## Testing
pmodel.load_weights(output_best_model)
pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])


print('best model monitored')
flog.write('best model monitored')

common_function_evaluation.evaluate(pmodel, test_generator,
                                    title = 'Boundary direction prediction',label='GM12878 test', curvefile='gm12878_boundary_prediction')

# pmodel.save(output_model_cnn)
# pmodel.save_weights(output_best_model)

model.save(output_model_cnn.replace('.h5','_1gpu.h5'))
model.save_weights(output_best_model.replace('.h5','_1gpu.h5'))



#########
flog.close()
















