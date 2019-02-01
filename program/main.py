#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:23:21 2018

@author: tat2016

To take a variant file from ICGC and generate input files and run the model to calculate loop probabilities

The program requires Homo_sapiens.GRCh37.75.dna.primary_assembly.fa in the same folder

Input:
The program requires 2 input files:
    + a simple somatic mutation file and
    + a structural variant file from ICGC (in .tsv format)


Run:
     python main.py ssm.tsv sv.tsv

Output: 2 folders
    + one folder contain input files, one for each patient, contains sequences of loops taking into consideration mutations from patients
    + another folder contains output loop probability prediction for each patient
-

"""

import multiprocessing as mp
import time
from keras.models import load_model
import os

import h5py

import numpy as np

from keras.utils import multi_gpu_model


# import variant_class
import boundary_class
import data_generator
import common_data_generation as cdg

import argparse



ref_genome_file = 'Homo_sapiens.GRCh37.75.dna.primary_assembly.fa'
cons_loop_file = 'loopDB/constitutive_loops.xlsx'

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
loop_model_file = 'model/loop_pred_4k_1gpu.h5'
ngpu = 1 # number of gpu to use, combine with os.environ above if necessary

nbr_job = 1  # number of parallel jobs when creating data for patients

# parameters to load data
batch_size = 24 * ngpu

# constants
segment_size = 4000
version = ''
nbr_feature = 10
rnn_len = 800



def makeOutputFolder():
    jobID = str(int(time.time()))

    # generate names and create output folders
    if not os.path.exists(jobID):
        os.makedirs(jobID)

    outputFolder = str(jobID) + '_output'
    inputFolder = str(jobID)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    return(jobID, inputFolder, outputFolder)


###########################################################
def get_prediction(model_file, data_file, isloop=True):
    """Run prediction for a data file"""

    params = {'dim': (segment_size, nbr_feature),
              'batch_size': batch_size,
              'n_channels': 1,
              'rnn_len': rnn_len}

    if isinstance(model_file, str):
        model = load_model(model_file)
    else:
        model = model_file

    rs = {}

    if isloop:
        generator = data_generator.DataGeneratorLoopSeq(data_file, None, shuffle=False, use_reverse=False, **params)
    else:
        generator = data_generator.DataGenerator(data_file, None, shuffle=False, use_reverse=False, **params)

    prediction = model.predict_generator(generator, verbose=1)
    # print(prediction, len(generator.ids_list))
    prediction = np.array(prediction).ravel()

    labels = generator.ids_list
    for i, k in enumerate(labels):
        if k in rs and abs(rs[k] - prediction[i]) > 0.00001:
            print('error in get_prediction, sample order is messed up')
        rs[k] = prediction[i]

    return (rs)


def output_prob(rs, file_name):
    """Output prediction """
    fout = open(file_name, 'w')
    for k, v in rs.items():
        fout.write('%s: \t %.10f\n' % (k, v))

    fout.close()


def process_eachpatient(loops, varlist, segment_size, ref_genome_file, output_file):
    """Generate data file for each patient

    """
    [looplist, _] = cdg.get_loop(loops, varlist, segment_size, isNormalize=False, isNonLoop=False, noLargeSV=False)

    looplist = [x for x in looplist if len(x.b1.variants) > 0 or len(x.b2.variants) > 0]
    if len(looplist) > 0:
        print(output_file)
        print('number of affected loops: %d' % (len(looplist)))

        if output_file:
            cdg.output_loop(looplist, ref_genome_file, segment_size, output_file, None)


def generate_patientdata(variants, loops, nbr_job, jobID):
    print('Creating input data for all patients...')

    # data file for each patient
    consitutive_loop_data_sampleid = lambda x: '%s/%s_loop_%s.mat' % (jobID, x, jobID)

    if nbr_job > 1:
        pool = mp.Pool(processes=nbr_job)
        result = [pool.apply_async(process_eachpatient, args=(loops, variants[k], segment_size,
                                                          ref_genome_file, consitutive_loop_data_sampleid(k))) for k in
                  variants]
        pool.close()
        pool.join()

    else:
        for k in variants:
            process_eachpatient(loops, variants[k], segment_size, ref_genome_file, consitutive_loop_data_sampleid(k))

    print('Done preparing data')


def makePrediction(loop_model_file, inputFolder, outputFolder):

    print('Loading model')
    model = load_model(loop_model_file)
    if ngpu > 1:
        model = multi_gpu_model(model, gpus=ngpu)

    patient_files = os.listdir(inputFolder)

    print('Running model on input data')
    for i in patient_files:

        input_file = os.path.join(inputFolder, i)

        dataset = h5py.File(input_file, 'r')

        print(input_file)
        if len(dataset.items()) == 0:
            continue

        output_file = os.path.join(outputFolder, i.replace('.mat', '.txt'))

        rs = get_prediction(model, input_file)

        output_prob(rs, output_file)

############################## Generating data

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-ssm', dest='ssm', default="", help='simple somatic mutation file')
    parser.add_argument('-sv', dest='sv', default="", help='structural variant file')
    parser.add_argument('-format', dest='format', default="tsv", help='file format (.bed) for TCGA WGS data, specify -ssm only (not -sv)')
    parser.add_argument('-loop', dest='loop_file', default=cons_loop_file, help='loop file')

    opts = parser.parse_args()

    # retrieve mutations file name from arguments
    inputF = opts.format #format file
    ssmFileName = opts.ssm
    svFileName = opts.sv

    loop_file = opts.loop_file

    ssmVariants = {}
    svVariants = {}

    print('Reading variants ...')

    if inputF == 'tsv':
        if ssmFileName:
            ssmVariants = cdg.getSSM(ssmFileName)
        if svFileName:
            svVariants = cdg.getSV(svFileName)
        variants = {**ssmVariants, **svVariants}

    elif inputF == 'bed':
        variants = cdg.read_ssm_bed_file(ssmFileName)



    if '.xlsx' in loop_file:
        loops = cdg.getLoopXLSX(loop_file)
    else:
        loops = boundary_class.read_loop(loop_file)



    #calculate probability for loops without mutations
    if not ssmFileName and not svFileName:
        print('No variants found, calculating loop probability for loops ...')

        output_seq_file = loop_file.replace(".xlsx", ".mat")
        [loops, _] = cdg.get_loop(loops, [], segment_size, isNormalize=False, isNonLoop=False, noLargeSV=False)
        cdg.output_loop(loops, ref_genome_file, segment_size, output_seq_file, None)

        rs = get_prediction(loop_model_file, output_seq_file)

        output_prob(rs, output_seq_file.replace(".mat", "_probability.txt"))

        os.remove(output_seq_file)

        return

    jobID, inputDataFolder, outputFolder = makeOutputFolder()

    generate_patientdata(variants, loops, nbr_job, jobID)

    makePrediction(loop_model_file, inputDataFolder, outputFolder)


main()
