#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:16:54 2018

@author: tat2016

CNN model to predict boundary from sequence only
"""


import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


from keras.models import Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization,\
                        Flatten,Input, Conv2D, MaxPooling2D, concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU


from keras.utils import multi_gpu_model

import data_generator
import roc_callback
import evaluation_function as ef

from datetime import datetime

def get_dilated_convnet(input, dilation_rate):
    x1 = Conv2D(filters=512, kernel_size=(5, 1), strides=(1, 1), dilation_rate=dilation_rate,
                         input_shape=(segment_size, 1, 1),
                         padding='same', kernel_initializer=initializer)(input)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(leaky)(x1)
    x1 = MaxPooling2D(pool_size=(segment_size - 16, 1), strides=(segment_size - 16, 1), padding='valid')(x1)
    x1 = Dropout(dropout)(x1)
    x1 = Flatten()(x1)

    return x1

##############################
epoch = 15

ngpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

batch_size = 12 * ngpu

existing_model = ''

segment_size = 4000 #length of boundaries

leaky = 0.2
dropout = 0.15

rnn_len = 0  # sequence length in input data, set to 0 to disable using RNN

version = ''

# cohort = 'mcf7'
# suffix = '_loocv'

# output_model_cnn = "model/boundary_sep_cnn_4k{}_test.h5".format(suffix + cohort)
# output_best_model = "model/boundary_sep_cnn_4k_best{}_test.h5".format(suffix + cohort)

# file_train_data = 'data/data_boundary_4k_train_leftout_{}.mat'.format(cohort)
# file_train_label = 'data/label_boundary_4k_train_leftout_{}.mat'.format(cohort)
#
# file_val_data = 'data/data_boundary_4k_val_leftout_{}.mat'.format(cohort)
# file_val_label = 'data/label_boundary_4k_val_leftout_{}.mat'.format(cohort)
#
# file_test_data1 = 'data/data_boundary_4k_test_type1_leftout_{}.mat'.format(cohort)
# file_test_label1 = 'data/label_boundary_4k_test_type1_leftout_{}.mat'.format(cohort)
# file_test_data2 = 'data/data_boundary_4k_test_type2_leftout_{}.mat'.format(cohort)
# file_test_label2 = 'data/label_boundary_4k_test_type2_leftout_{}.mat'.format(cohort)
# file_test_data3 = 'data/data_boundary_4k_test_type3_leftout_{}.mat'.format(cohort)
# file_test_label3 = 'data/label_boundary_4k_test_type3_leftout_{}.mat'.format(cohort)


# file_train_data = 'data/data_boundary_4k_train_ruan.mat'
# file_train_label = 'data/label_boundary_4k_train_ruan.mat'
# # file_train_data = 'data/data_boundary_4k_val_ruan.mat'
# # file_train_label = 'data/label_boundary_4k_val_ruan.mat'
#
# file_val_data = 'data/data_boundary_4k_val_ruan.mat'
# file_val_label = 'data/label_boundary_4k_val_ruan.mat'
#
# file_test_data = 'data/data_boundary_4k_test_ruan.mat'
# file_test_label = 'data/label_boundary_4k_test_ruan.mat'


output_model_cnn = "model/boundary_sep_cnn_4k_test141.h5"
output_best_model = "model/boundary_sep_cnn_4k_best_test141.h5"

file_train_data = 'data/data_boundary_4k_traintest141_ruan.mat'
file_train_label = 'data/label_boundary_4k_traintest141_ruan.mat'

file_val_data = 'data/data_boundary_4k_valtest141_ruan.mat'
file_val_label = 'data/label_boundary_4k_valtest141_ruan.mat'

file_test_data1 = 'data/data_boundary_4k_testtest141_type1_ruan.mat'
file_test_label1 = 'data/label_boundary_4k_testtest141_type1_ruan.mat'
file_test_data2 = 'data/data_boundary_4k_testtest141_type2_ruan.mat'
file_test_label2 = 'data/label_boundary_4k_testtest141_type2_ruan.mat'
file_test_data3 = 'data/data_boundary_4k_testtest141_type3_ruan.mat'
file_test_label3 = 'data/label_boundary_4k_testtest141_type3_ruan.mat'



n_letters = 5 # len('ACGTN')

nbr_feature = n_letters# number of features

script_name = sys.argv[0].replace('.py','').split('/')[-1]

log_file = "log/%s_log_%s.txt" % (script_name, datetime.now().strftime('%H_%M_%d_%m_%Y'))

flog = open(log_file,"w")

flog.write('existing model:{}\n'.format(existing_model))
flog.write('leaky:{}, dropout:{}, rnnlen: {}, segment_size:{}\n'.format(leaky, dropout, rnn_len, segment_size))
flog.write('version:{}, existing model:{}\n'.format(version, existing_model))
flog.write('train data: {}, {}\n'.format(file_train_data, file_train_label))
flog.flush()

########## Loading data
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len}

train_generator = data_generator.DataGenerator(file_train_data, file_train_label, shuffle=True, **params)
val_generator = data_generator.DataGenerator(file_val_data, file_val_label, shuffle=False, **params) #set shuffle=False to calculate AUC
test_generator1 = data_generator.DataGenerator(file_test_data1, file_test_label1, shuffle=False, use_reverse=False, **params) #set shuffle=False to calculate AUC
test_generator2 = data_generator.DataGenerator(file_test_data2, file_test_label2, shuffle=False, use_reverse=False, **params) #set shuffle=False to calculate AUC
test_generator3 = data_generator.DataGenerator(file_test_data3, file_test_label3, shuffle=False, use_reverse=False, **params) #set shuffle=False to calculate AUC


########## Build network
seq_input = Input(shape=(segment_size,nbr_feature,1))


initializer = 'glorot_uniform'

x = Conv2D(filters=256, kernel_size = (17,5), strides=(1,1), input_shape=(segment_size, nbr_feature,1),
           padding='valid', kernel_initializer = initializer)(seq_input)

x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)



x1 = get_dilated_convnet(x, dilation_rate=1)
x2 = get_dilated_convnet(x, dilation_rate=3)
x3 = get_dilated_convnet(x, dilation_rate=7)

x = concatenate([x1,x2,x3])


x = Dense(units=256)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)

x = Dropout(dropout)(x)

x = Dense(units=128)(x)
x = BatchNormalization()(x)
x = LeakyReLU(leaky)(x)
x = Dropout(dropout)(x)

x = Dense(units=1)(x)

predictions = Activation('sigmoid')(x)

inputs = [seq_input]

model = Model(inputs= inputs , outputs=predictions)

model.summary()

# model.compile(loss='binary_crossentropy', optimizer='rmsprop',
#               metrics=['accuracy'])

if ngpu > 1:
    pmodel = multi_gpu_model(model, gpus=ngpu)
else:
    pmodel = model

pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy', ef.fbeta_score])

#pmodel.summary()


if existing_model:
    pmodel.load_weights(existing_model)

################ Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
checkpoint = ModelCheckpoint(output_best_model, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

roc = roc_callback.ROC_Callback(val_generator=val_generator, fout=flog)
callback_list = [checkpoint, early_stopping]


pmodel.fit_generator(generator=train_generator, epochs=epoch, shuffle=True,
      validation_data=val_generator, use_multiprocessing=False, workers=ngpu,
      callbacks=callback_list, verbose=1)



################ Test with best trained models
pmodel.load_weights(output_best_model)
# Compile model (required to make predictions)
pmodel.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy', ef.fbeta_score])


flog.write('best model monitored')


ef.evaluate(pmodel, test_generator1, flog=flog, name=file_test_label1, output_file=file_test_label1.replace('.mat','_sepcnn_output.txt'))
ef.evaluate(pmodel, test_generator2, flog=flog, name=file_test_label2, output_file=file_test_label2.replace('.mat','_sepcnn_output.txt'))
ef.evaluate(pmodel, test_generator3, flog=flog, name=file_test_label3, output_file=file_test_label3.replace('.mat','_sepcnn_output.txt'))


pmodel.save(output_model_cnn)
pmodel.save_weights(output_best_model)

model.save(output_model_cnn.replace('.h5','_1gpu.h5'))
model.save_weights(output_best_model.replace('.h5','_1gpu.h5'))


#########
flog.close()




















