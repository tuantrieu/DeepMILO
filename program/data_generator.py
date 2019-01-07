'''
Data generation for input as sequences and output as boundaries
'''
import numpy as np
import keras
import h5py
import re
import random


class DataGenerator(keras.utils.Sequence):
    'Generates data for Boundary prediction'

    def __init__(self, data_file, label_file, batch_size=32, dim=(1000, 10), n_channels=1,
                 shuffle=True, rnn_len=0, use_reverse=True, rnn_only=False):
        '''
        shuffle: shuffle data for training, in validation and testing set it to False
        rnn_len: len of the RNN layer
        use_reverse: whether to consider the reverse strand (with label ends with _2), can be used during testing
        rnn_only: return data for RNN only
        '''
        'Initialization'

        self.data_file = data_file
        self.label_file = label_file

        self.dim = dim
        self.batch_size = batch_size

        self.rnn_len = rnn_len
        self.rnn_only = rnn_only

        self.n_channels = n_channels

        self.shuffle = shuffle

        self.use_reverse = use_reverse

        self.ids_list = []

        # self.indexes = []


        h5_data = None
        try:
            h5_data = h5py.File(data_file, 'r')
            for k, v in h5_data.items():
                if not self.use_reverse:
                    
                    tmpk = k
                    if re.search('\.[0-9]+$', k): #take care of the case .[0-9] is appended
                        tmpk = re.sub('\.[0-9]+$','',k)
                        #print(tmpk)
                        
                    if tmpk.endswith("_1"):  # only take the forward strand
                        self.ids_list.append(k)

                else:
                    self.ids_list.append(k)

            le = int(np.floor(len(self.ids_list) / self.batch_size))
            #taking care of the case when number of samples < batch_size: randomly add samples to make a full batch size
            if le * self.batch_size < len(self.ids_list):
                
                shortage = (le + 1) * self.batch_size  - len(self.ids_list) # add this number of samples to 

                i = 1
                while len(self.ids_list) * i < shortage:
                    i += 1
                    
                add = random.sample(self.ids_list * i, shortage) # randomly sample
                self.ids_list = self.ids_list  + add
                
        finally:
            if h5_data is not None:
                h5_data.close()

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        le = int(np.floor(len(self.ids_list) / self.batch_size))
        
        return (le)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        ids_list_temp = [self.ids_list[k] for k in indexes]

        # Generate data
        if self.label_file:
            X, y = self.__data_generation(ids_list_temp)
        else:
            X = self.__data_generation(ids_list_temp)

        # if generate additional data for RNN to run
        # RNN doesn't take the full length for computational efficiency purpose
        # if return data for RNN, the two alleles are returned separately
        if self.rnn_len > 0:
            seq_len = self.dim[0]
            gru_start = int(seq_len / 2 - self.rnn_len / 2)
            gru_end = int(seq_len / 2 + self.rnn_len / 2)
            
            if self.label_file:
                
                if not self.rnn_only:
                    return ([X, X[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5),
                             X[:, gru_start:gru_end, 5:10].reshape(self.batch_size, -1, 5)], y)
                else:
                    return (X[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5), y)
            
            else:
                if not self.rnn_only:
                    return ([X, X[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5),
                             X[:, gru_start:gru_end, 5:10].reshape(self.batch_size, -1, 5)])
                else:
                    return (X[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5))
        else:
            if self.label_file:
                return (X, y)
            else:
                return(X)

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.ids_list))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_list_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.n_channels > 0:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.dim))

        # y = np.empty((self.batch_size), dtype=int)
        y = []

        h5_data = None
        h5_label = None

        try:
            h5_data = h5py.File(self.data_file, 'r')
            
            if self.label_file:
                h5_label = h5py.File(self.label_file, 'r')

            # Generate data
            for i, ID in enumerate(ids_list_temp):
                # Store sample
                dt = h5_data[ID][:]
                if self.n_channels > 0:
                    X[i,] = dt.reshape(*self.dim, self.n_channels)
                else:
                    X[i,] = dt.reshape(*self.dim)
                
                if self.label_file:
                    lb = h5_label[ID]
                    y.append(lb.value)
            
            if self.label_file:
                y = np.stack(y)

        finally:
            if h5_data is not None:
                h5_data.close()
            if h5_label is not None:
                h5_label.close()
                
        if self.label_file:
            return (X, y)
        else:
            return (X)






class DataGeneratorLoopSeq(keras.utils.Sequence):
    'Generates sequence data to predict loop'

    def __init__(self, data_file, label_file, batch_size=32, dim=(1000, 10), n_channels=1,
                 shuffle=True, rnn_len=0, use_reverse=True):
        '''
        rnn_len: len of the RNN layer
        use_reverse: whether to consider the reverse strand (with label ends with _2), can be used during testing
        '''
        'Initialization'

        self.data_file = data_file
        self.label_file = label_file

        self.dim = dim
        self.batch_size = batch_size

        self.rnn_len = rnn_len

        self.n_channels = n_channels

        self.shuffle = shuffle

        self.use_reverse = use_reverse

        self.ids_list = []

        # self.indexes = []

        h5_data = None
        try:
            h5_data = h5py.File(data_file, 'r')
            for k, v in h5_data.items():
                if not self.use_reverse:
                    tmpk = k
                    if re.search('\.[0-9]+$', k): #take care of the case .[0-9] is appended
                        tmpk = re.sub('\.[0-9]+$','',k)
                        #print(tmpk)
                    
                    #print(tmpk, tmpk.endswith("_1") )
                    if tmpk.endswith("_1"):  # only take the forward strand
                        self.ids_list.append(k)

                else:
                    self.ids_list.append(k)
                
                

            le = int(np.floor(len(self.ids_list) / self.batch_size))
            #taking care of the case when number of samples < batch_size: randomly add samples to make a full batch size
            if le * self.batch_size < len(self.ids_list):
                
                shortage = (le + 1) * self.batch_size  - len(self.ids_list) # add this number of samples to 
                i = 1
                while len(self.ids_list) * i < shortage:
                    i += 1
                    
                add = random.sample(self.ids_list * i, shortage) # randomly sample
                
                self.ids_list = self.ids_list  + add
                
        finally:
            if h5_data is not None:
                h5_data.close()

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        le = int(np.floor(len(self.ids_list) / self.batch_size))
            
        return (le)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: min((index + 1) * self.batch_size, len(self.ids_list) )]

        # Find list of IDs
        ids_list_temp = [self.ids_list[k] for k in indexes]

        # Generate data
        X1, X2, y = self.__data_generation(ids_list_temp)

        # if generate additional data for RNN to run
        # RNN doesn't take the full length for computational efficiency purpose
        # if return data for RNN, the two alleles are returned separately
        if self.rnn_len > 0:
            seq_len = self.dim[0]
            gru_start = int(seq_len / 2 - self.rnn_len / 2)
            gru_end = int(seq_len / 2 + self.rnn_len / 2)

            return ([X1, X1[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5),
                     X1[:, gru_start:gru_end, 5:10].reshape(self.batch_size, -1, 5),
                     X2, X2[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5),
                     X2[:, gru_start:gru_end, 5:10].reshape(self.batch_size, -1, 5)],
                    y)
        else:
            return ([X1, X2], y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.ids_list))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_list_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.n_channels > 0:
            X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
            X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X1 = np.empty((self.batch_size, *self.dim))
            X2 = np.empty((self.batch_size, *self.dim))

        # y = np.empty((self.batch_size), dtype=int)
        y = []

        h5_data = None
        h5_label = None

        try:
            h5_data = h5py.File(self.data_file, 'r')
            
            if self.label_file:
                h5_label = h5py.File(self.label_file, 'r')

            # Generate data
            for i, ID in enumerate(ids_list_temp):

                dt = h5_data[ID][:]
                
                # data shape is (1000, 20), X1 will be (1000,10), X2: (1000,10)
                half_nbr_seq = int(dt.shape[1] / 2)
                
                dt1 = dt[:, : half_nbr_seq]
                dt2 = dt[:, half_nbr_seq:]

                if self.n_channels > 0:
                    X1[i,] = dt1.reshape(*self.dim, self.n_channels)
                    X2[i,] = dt2.reshape(*self.dim, self.n_channels)
                else:
                    X1[i,] = dt1.reshape(*self.dim)
                    X2[i,] = dt2.reshape(*self.dim)
                
                if h5_label:
                    lb = h5_label[ID]
                    y.append(lb.value)
            if self.label_file:
                y = np.stack(y)

        finally:
            if h5_data is not None:
                h5_data.close()
            if h5_label is not None:
                h5_label.close()

        return (X1, X2, y)




#class DataGeneratorSeqAndState(keras.utils.Sequence):
#    '''Generates data for Keras
#    
#    two input: sequence and chromatin state
#    one output: boundary or not
#    '''
#
#    def __init__(self, data_file, boundary_label_file, chromatin_state_label_file, batch_size=32, dim=(1000, 10),
#                 n_channels=1,
#                 shuffle=True, rnn_len=0, use_reverse=True):
#        '''
#        rnn_len: len of the RNN layer
#        use_reverse: whether to consider the reverse strand (with label ends with _2), may be considred for testing
#        '''
#        'Initialization'
#
#        self.data_file = data_file
#
#        self.boundary_label_file = boundary_label_file
#        self.chromatin_state_label_file = chromatin_state_label_file
#
#        self.dim = dim
#        self.batch_size = batch_size
#
#        self.rnn_len = rnn_len
#
#        self.n_channels = n_channels
#
#        self.shuffle = shuffle
#
#        self.use_reverse = use_reverse
#
#        self.ids_list = []
#
#        h5_label = None
#        try:
#            h5_label = h5py.File(boundary_label_file, 'r')
#            for k, v in h5_label.items():
#                if not self.use_reverse:
#                    if k.endswith("_1"):  # only take the forward strand
#                        self.ids_list.append(k)
#                else:
#                    self.ids_list.append(k)
#        finally:
#            if h5_label is not None:
#                h5_label.close()
#
#        self.on_epoch_end()
#
#    def __len__(self):
#        'Denotes the number of batches per epoch'
#        return int(np.floor(len(self.ids_list) / self.batch_size))
#
#    def __getitem__(self, index):
#        'Generate one batch of data'
#        # Generate indexes of the batch
#        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#        # Find list of IDs
#        ids_list_temp = [self.ids_list[k] for k in indexes]
#
#        # Generate data
#        X, X_states, y_boundary = self.__data_generation(ids_list_temp)
#
#        # if generate additional data for RNN to run
#        # RNN doesn't take the full length for computational efficiency purpose
#        # if return data for RNN, the two alleles are returned separately
#        if self.rnn_len > 0:
#            seq_len = self.dim[0]
#            gru_start = int(seq_len / 2 - self.rnn_len / 2)
#            gru_end = int(seq_len / 2 + self.rnn_len / 2)
#
#            return ([X, X[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5),
#                     X[:, gru_start:gru_end, 5:10].reshape(self.batch_size, -1, 5)], X_states, y_boundary)
#        else:
#            return ({'seq': X, 'state': X_states}, y_boundary)
#
#    def on_epoch_end(self):
#        'Updates indexes after each epoch'
#        self.indexes = np.arange(len(self.ids_list))
#        if self.shuffle == True:
#            np.random.shuffle(self.indexes)
#
#    def __data_generation(self, ids_list_temp):
#        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
#        # Initialization
#        if self.n_channels > 0:
#            X = np.empty((self.batch_size, *self.dim, self.n_channels))
#        else:
#            X = np.empty((self.batch_size, *self.dim))
#
#        # y = np.empty((self.batch_size), dtype=int)
#        y_boundary = []
#        X_states = []
#
#        h5_data = None
#        h5_boundary_label = None
#        h5_chromatin_state_label = None
#
#        try:
#            h5_data = h5py.File(self.data_file, 'r')
#            h5_boundary_label = h5py.File(self.boundary_label_file, 'r')
#            h5_chromatin_state_label = h5py.File(self.chromatin_state_label_file, 'r')
#
#            # Generate data
#            for i, ID in enumerate(ids_list_temp):
#                # Store sample
#                dt = h5_data[ID][:]
#                if self.n_channels > 0:
#                    X[i,] = dt.reshape(*self.dim, self.n_channels)
#                else:
#                    X[i,] = dt.reshape(*self.dim)
#
#                y_boundary.append(h5_boundary_label[ID].value)
#
#                # state_ID = re.sub('_[12]$','',ID)
#                X_states.append(h5_chromatin_state_label[ID])
#
#            y_boundary = np.stack(y_boundary)
#            X_states = np.stack(X_states)
#
#        finally:
#            if h5_data is not None:
#                h5_data.close()
#
#            if h5_boundary_label is not None:
#                h5_boundary_label.close()
#
#            if h5_chromatin_state_label is not None:
#                h5_chromatin_state_label.close()
#
#        return (X, X_states, y_boundary)
#
#
#class DataGeneratorBoundaryAndState(keras.utils.Sequence):
#    '''Generates data for Keras
#    
#    one input: sequences
#    two outputs: boundaries and chromatins states
#
#    '''
#
#    def __init__(self, data_file, boundary_label_file, chromatin_state_label_file, batch_size=32, dim=(1000, 10),
#                 n_channels=1,
#                 shuffle=True, rnn_len=0, use_reverse=True):
#        '''
#        rnn_len: len of the RNN layer
#        use_reverse: whether to consider the reverse strand (with label ends with _2), can be used during testing
#        '''
#        'Initialization'
#
#        self.data_file = data_file
#
#        self.boundary_label_file = boundary_label_file
#        self.chromatin_state_label_file = chromatin_state_label_file
#
#        self.dim = dim
#        self.batch_size = batch_size
#
#        self.rnn_len = rnn_len
#
#        self.n_channels = n_channels
#
#        self.shuffle = shuffle
#
#        self.use_reverse = use_reverse
#
#        self.ids_list = []
#
#        h5_label = None
#        try:
#            h5_label = h5py.File(boundary_label_file, 'r')
#            for k, v in h5_label.items():
#                if not self.use_reverse:
#                    if k.endswith("_1"):  # only take the forward strand
#                        self.ids_list.append(k)
#                else:
#                    self.ids_list.append(k)
#        finally:
#            if h5_label is not None:
#                h5_label.close()
#
#        self.on_epoch_end()
#
#    def __len__(self):
#        'Denotes the number of batches per epoch'
#        return int(np.floor(len(self.ids_list) / self.batch_size))
#
#    def __getitem__(self, index):
#        'Generate one batch of data'
#        # Generate indexes of the batch
#        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#        # Find list of IDs
#        ids_list_temp = [self.ids_list[k] for k in indexes]
#
#        # Generate data
#        X, y_boundary, y_states = self.__data_generation(ids_list_temp)
#
#        # if generate additional data for RNN to run
#        # RNN doesn't take the full length for computational efficiency purpose
#        # if return data for RNN, the two alleles are returned separately
#        if self.rnn_len > 0:
#            seq_len = self.dim[0]
#            gru_start = int(seq_len / 2 - self.rnn_len / 2)
#            gru_end = int(seq_len / 2 + self.rnn_len / 2)
#
#            return ([X, X[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5),
#                     X[:, gru_start:gru_end, 5:10].reshape(self.batch_size, -1, 5)], y_boundary, y_states)
#        else:
#            return (X, {'boundary': y_boundary, 'state': y_states})
#
#    def on_epoch_end(self):
#        'Updates indexes after each epoch'
#        self.indexes = np.arange(len(self.ids_list))
#        if self.shuffle == True:
#            np.random.shuffle(self.indexes)
#
#    def __data_generation(self, ids_list_temp):
#        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
#        # Initialization
#        if self.n_channels > 0:
#            X = np.empty((self.batch_size, *self.dim, self.n_channels))
#        else:
#            X = np.empty((self.batch_size, *self.dim))
#
#        # y = np.empty((self.batch_size), dtype=int)
#        y_boundary = []
#        y_states = []
#
#        h5_data = None
#        h5_boundary_label = None
#        h5_chromatin_state_label = None
#
#        try:
#            h5_data = h5py.File(self.data_file, 'r')
#            h5_boundary_label = h5py.File(self.boundary_label_file, 'r')
#            h5_chromatin_state_label = h5py.File(self.chromatin_state_label_file, 'r')
#
#            # Generate data
#            for i, ID in enumerate(ids_list_temp):
#                # Store sample
#                dt = h5_data[ID][:]
#                if self.n_channels > 0:
#                    X[i,] = dt.reshape(*self.dim, self.n_channels)
#                else:
#                    X[i,] = dt.reshape(*self.dim)
#
#                y_boundary.append(h5_boundary_label[ID].value)
#
#                # state_ID = re.sub('_[12]$','',ID)
#                y_states.append(h5_chromatin_state_label[ID])
#
#            y_boundary = np.stack(y_boundary)
#            y_states = np.stack(y_states)
#
#        finally:
#            if h5_data is not None:
#                h5_data.close()
#
#            if h5_boundary_label is not None:
#                h5_boundary_label.close()
#
#            if h5_chromatin_state_label is not None:
#                h5_chromatin_state_label.close()
#
#        return (X, y_boundary, y_states)
