#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:00:51 2018

@author: tat2016

To generate data for loop prediction from boundaries
"""


import numpy as np
from Bio import SeqIO

from region_class import Region
from boundary_class import Loop

import boundary_class
import variant_class


import os
import random

import common_data_generation as cdg

random.seed(1)

#from multiprocessing import Pool

#############


ref_genome_file = "/Users/tat2016/data/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa"
gm12878_loop_file = "/Users/tat2016/Box Sync/Research/Data/CHIA_PET/Heidari.GM12878.Rad21.mango.interactions.FDR0.2.mango.allCC.txt"

if not os.path.isfile(ref_genome_file):
    ref_genome_file = "/athena/khuranalab/scratch/tat2016/data/deep_seq/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa"
    gm12878_loop_file = "/athena/khuranalab/scratch/tat2016/data/deep_seq/Heidari.GM12878.Rad21.mango.interactions.FDR0.2.mango.allCC.txt"
    
variant_file = "/Users/tat2016/Box Sync/Research/Data/Variants/1000Genomes/gm12878_variants.vcf"

ctcfFile = '/Users/tat2016/Box Sync/Research/Data/ctcf_positions.tsv'

segment_size = 4000
half_segment = int(segment_size/2)


version = "" 

output_file_train_data = 'data_loop_%dk_train%s.mat' % (segment_size/1000, version)
output_file_train_label = 'label_loop_%dk_train%s.mat' % (segment_size/1000, version)

output_file_val_data = 'data_loop_%dk_val%s.mat' % (segment_size/1000, version)
output_file_val_label = 'label_loop_%dk_val%s.mat' % (segment_size/1000, version)

output_file_test_data = 'data_loop_%dk_test%s.mat' % (segment_size/1000, version)
output_file_test_label = 'label_loop_%dk_test%s.mat' % (segment_size/1000, version)




'''Check if any loop out of range of chromosomes'''
#chromList = cdg.getChromlist() 
#
#loops = cdg.getLoopFromLabelFile(output_file_train_label)
#
#for x in loops:
#    if x.b1.end > len(chromList[x.b1.chrom]) or x.b2.end > len(chromList[x.b2.chrom]):
#        print(str(x))
''''''



val_chrom = ['chr16']
test_chrom = ['chr8', 'chr7']

sample_name = 'NA12878'

################# Loading chromatin state data



################## positive regions, boundary regions

#imp.reload(variant_class)


def getPosLoop(loopFile, variantFile, ctcfFile, segment_size):
    '''pass'''
    
    if isinstance(variantFile, str):
        variants = variant_class.extract_variants(variantFile, sample_name, all_variant = True)
    else:
        variants = variantFile
    
    [norm_looplist, _] = cdg.get_loop(loopFile, variants, segment_size, 
                                        ctcfFile = ctcfFile, nonloopType = 0, isNonLoop=False)

    
    print('number of loop with convergent CTCF: %d/%d' % 
          (len([x for x in norm_looplist if x.b1.ctcf_dir in [2,3] and x.b2.ctcf_dir in [1,3]]), len(norm_looplist)))
    
    return(norm_looplist)


def splitData(loops, testChrom, valChrom):
    trainData = [x for x in loops if not x.b1.chrom in (testChrom + valChrom)]
    testData = [x for x in loops if x.b1.chrom in testChrom]
    valData = [x for x in loops if x.b1.chrom in valChrom]
    
    return(trainData, testData, valData)

def getNegLoop(posLoop, variants, ctcfFile, segment_size):
    
    
    nonloopsConv = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 1, segment_size)
    
    print('number of neg. loop with convergent CTCF: %d/%d' % 
          (len([x for x in nonloopsConv if x.b1.ctcf_dir in [2,3] and x.b2.ctcf_dir in [1,3]]), len(nonloopsConv)))
    
    nonloopsTandem = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 2, segment_size)
    
    nonloopsDiver = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 5, segment_size)
    
    print('number of neg. loop with divergent CTCF: %d/%d' % 
          (len([x for x in nonloopsDiver if x.b1.ctcf_dir in [1,3] and x.b2.ctcf_dir in [2,3]]), len(nonloopsDiver)))
    
    
    nonloopsOneFakeCTCF = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 3, segment_size)
    
    
    nonloopsOneFakeNoCTCF = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 4, segment_size)
    
    return(nonloopsConv, nonloopsTandem, nonloopsDiver, nonloopsOneFakeCTCF, nonloopsOneFakeNoCTCF)
    


variants = variant_class.extract_variants(variant_file, sample_name, all_variant = True)   

posLoop  = getPosLoop(gm12878_loop_file, variants, ctcfFile, segment_size)

print('number of positive loops: %d' % (len(posLoop)))

''''''


nonloopsConv = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 1, segment_size)

print('number of neg. loop with convergent CTCF: %d/%d' % 
      (len([x for x in nonloopsConv if x.b1.ctcf_dir in [2,3] and x.b2.ctcf_dir in [1,3]]), len(nonloopsConv)))

nonloopsTandem = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 2, segment_size)

nonloopsDiver = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 5, segment_size)

print('number of neg. loop with divergent CTCF: %d/%d' % 
      (len([x for x in nonloopsDiver if x.b1.ctcf_dir in [1,3] and x.b2.ctcf_dir in [2,3]]), len(nonloopsDiver)))


nonloopsOneFakeCTCF = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 3, segment_size)

#import imp

nonloopsOneFakeNoCTCF = cdg.get_nonloop(posLoop, variants, True, ctcfFile, 4, segment_size)

''''''


#nonloopsConv, nonloopsTandem, nonloopsDiver, nonloopsOneFakeCTCF, nonloopsOneFakeNoCTCF = getNegLoop(posLoop, variants, ctcfFile, segment_size)

#'''Sanity check'''
#print(len([x for x in nonloopsTandem if x.b1.ctcf_dir == x.b2.ctcf_dir and x.b1.ctcf_dir != 3]), len(nonloopsTandem))
#print(len([x for x in nonloopsTandem if x in posLoop]))
#
#print(len([x for x in nonloopsDiver if x.b1.ctcf_dir in [1,3] and x.b2.ctcf_dir in [2,3] and x.b1.ctcf_dir != x.b2.ctcf_dir]), len(nonloopsDiver))
#print(len([x for x in nonloopsDiver if x in posLoop]))
#
#print(len([x for x in nonloopsOneFakeNoCTCF if x.b1.ctcf_dir * x.b2.ctcf_dir == 0 and x.b1.ctcf_dir + x.b2.ctcf_dir > 0]), len(nonloopsOneFakeNoCTCF))
#print(len([x for x in nonloopsOneFakeNoCTCF if x in posLoop]))
#
#
#allBoundaries = [x.b1 for x in posLoop] + [x.b2 for x in posLoop]
#print(len([x for x in nonloopsOneFakeCTCF if x.b1 in allBoundaries and x.b2.ctcf_dir > 0 
#           and not x.b2 in allBoundaries ]), len(nonloopsOneFakeCTCF))
#    
#print(len([x for x in nonloopsOneFakeCTCF if x in posLoop]))
#
#print(len([x for x in nonloopsConv if x.b1.ctcf_dir in [2,3] and x.b2.ctcf_dir in [1,3]]), len(nonloopsConv))
#print(len([x for x in nonloopsConv if x in posLoop]))
#
#'''Done sanity check'''





random.shuffle(posLoop), random.shuffle(nonloopsConv), random.shuffle(nonloopsTandem), 
random.shuffle(nonloopsDiver), random.shuffle(nonloopsOneFakeCTCF), random.shuffle(nonloopsOneFakeNoCTCF)

posLoopTrain, posLoopTest, posLoopVal = splitData(posLoop, test_chrom, val_chrom)

nonloopsConvTrain, nonloopsConvTest, nonloopsConvVal = splitData(nonloopsConv, test_chrom, val_chrom)
nonloopsTandemTrain, nonloopsTandemTest, nonloopsTandemVal = splitData(nonloopsTandem, test_chrom, val_chrom)
nonloopsDiverTrain, nonloopsDiverTest, nonloopsDiverVal = splitData(nonloopsDiver, test_chrom, val_chrom)
nonloopsOneFakeCTCFTrain, nonloopsOneFakeCTCFTest, nonloopsOneFakeCTCFVal = splitData(nonloopsOneFakeCTCF, test_chrom, val_chrom)
nonloopsOneFakeNoCTCFTrain, nonloopsOneFakeNoCTCFTest, nonloopsOneFakeNoCTCFVal = splitData(nonloopsOneFakeNoCTCF, test_chrom, val_chrom)


posTrainLen, posTestLen, posValLen = len(posLoopTrain), len(posLoopTest), len(posLoopVal)

negTrain = nonloopsConvTrain[: int(0.5 * posTrainLen)] + nonloopsTandemTrain[: int(0.1 * posTrainLen)]\
                + nonloopsDiverTrain[: int(0.1 * posTrainLen)] + nonloopsOneFakeCTCFTrain[: int(0.2 * posTrainLen)]\
                + nonloopsOneFakeNoCTCFTrain[: int(0.1 * posTrainLen)]\

negTest = nonloopsConvTest[: int(0.5 * posTestLen)] + nonloopsTandemTest[: int(0.1 * posTestLen)]\
                + nonloopsDiverTest[: int(0.1 * posTestLen)] + nonloopsOneFakeCTCFTest[: int(0.2 * posTestLen)]\
                + nonloopsOneFakeNoCTCFTest[: int(0.1 * posTestLen)]\

negVal = nonloopsConvVal[: int(0.5 * posValLen)] + nonloopsTandemVal[: int(0.1 * posValLen)]\
                + nonloopsDiverVal[: int(0.1 * posValLen)] + nonloopsOneFakeCTCFVal[: int(0.2 * posValLen)]\
                + nonloopsOneFakeNoCTCFVal[: int(0.1 * posValLen)]\



cdg.output_loop(posLoopTrain + negTrain, ref_genome_file, segment_size, 
                              output_file_train_data, output_file_train_label)


cdg.output_loop(posLoopVal + negVal, ref_genome_file, segment_size, 
                              output_file_val_data, output_file_val_label)

cdg.output_loop(posLoopTest + negTest, ref_genome_file, segment_size, 
                              output_file_test_data, output_file_test_label)




'''Datasets with all negative samples with convergent boundaries for testing word2Vec model'''
trainLen = min(posTrainLen, len(nonloopsConvTrain))
testLen = min(posTestLen, len(nonloopsConvTest))
valLen = min(posValLen, len(nonloopsConvVal))

negTrain = nonloopsConvTrain[: trainLen]
negTest = nonloopsConvTest[: testLen]
negVal = nonloopsConvVal[: valLen]


version = "NegConv4Train" 

output_file_train_data = 'data_loop_%dk_train%s.mat' % (segment_size/1000, version)
output_file_train_label = 'label_loop_%dk_train%s.mat' % (segment_size/1000, version)

output_file_val_data = 'data_loop_%dk_val%s.mat' % (segment_size/1000, version)
output_file_val_label = 'label_loop_%dk_val%s.mat' % (segment_size/1000, version)

output_file_test_data = 'data_loop_%dk_test%s.mat' % (segment_size/1000, version)
output_file_test_label = 'label_loop_%dk_test%s.mat' % (segment_size/1000, version)

cdg.output_loop(posLoopTrain + negTrain, ref_genome_file, segment_size, 
                              output_file_train_data, output_file_train_label)


cdg.output_loop(posLoopVal + negVal, ref_genome_file, segment_size, 
                              output_file_val_data, output_file_val_label)

cdg.output_loop(posLoopTest + negTest, ref_genome_file, segment_size, 
                              output_file_test_data, output_file_test_label)



#''' Full datasets '''
#
#version = "FullUnbalanced" 
#
#output_file_train_data = 'data_loop_%dk_train%s.mat' % (segment_size/1000, version)
#output_file_train_label = 'label_loop_%dk_train%s.mat' % (segment_size/1000, version)
#
#output_file_val_data = 'data_loop_%dk_val%s.mat' % (segment_size/1000, version)
#output_file_val_label = 'label_loop_%dk_val%s.mat' % (segment_size/1000, version)
#
#output_file_test_data = 'data_loop_%dk_test%s.mat' % (segment_size/1000, version)
#output_file_test_label = 'label_loop_%dk_test%s.mat' % (segment_size/1000, version)
#
#
#negTrain = nonloopsConvTrain[: int(posTrainLen)] + nonloopsTandemTrain[: int(0.2 * posTrainLen)]\
#                + nonloopsDiverTrain[: int(0.2 * posTrainLen)] + nonloopsOneFakeCTCFTrain[: int(0.4 * posTrainLen)]\
#                + nonloopsOneFakeNoCTCFTrain[: int(0.2 * posTrainLen)]\
#
#negTest = nonloopsConvTest[: int(posTestLen)] + nonloopsTandemTest[: int(0.2 * posTestLen)]\
#                + nonloopsDiverTest[: int(0.2 * posTestLen)] + nonloopsOneFakeCTCFTest[: int(0.4 * posTestLen)]\
#                + nonloopsOneFakeNoCTCFTest[: int(0.2 * posTestLen)]\
#
#negVal = nonloopsConvVal[: int(posValLen)] + nonloopsTandemVal[: int(0.2 * posValLen)]\
#                + nonloopsDiverVal[: int(0.2 * posValLen)] + nonloopsOneFakeCTCFVal[: int(0.4 * posValLen)]\
#                + nonloopsOneFakeNoCTCFVal[: int(0.2 * posValLen)]\
#
#
#
#cdg.output_loop(posLoopTrain + negTrain, ref_genome_file, segment_size, 
#                              output_file_train_data, output_file_train_label)
#
#
#cdg.output_loop(posLoopVal + negVal, ref_genome_file, segment_size, 
#                              output_file_val_data, output_file_val_label)
#
#cdg.output_loop(posLoopTest + negTest, ref_genome_file, segment_size, 
#                              output_file_test_data, output_file_test_label)






''' '''





'''Fake loops with convergent boundaries'''
output_file_nonloop_data = 'data_nonloop_%dk%s.mat' % (segment_size/1000, 'negConv')
output_file_nonloop_label = 'label_nonloop_%dk%s.mat' % (segment_size/1000, 'negConv')

len(nonloopsConvTest), len(posLoopTest)
cdg.output_loop(posLoopTest[:len(nonloopsConvTest)] + nonloopsConvTest, ref_genome_file, segment_size, 
                              output_file_nonloop_data, output_file_nonloop_label)






output_file_nonloop_data = 'data_nonloop_%dk%s.mat' % (segment_size/1000, 'negTandem')
output_file_nonloop_label = 'label_nonloop_%dk%s.mat' % (segment_size/1000, 'negTandem')

len(nonloopsTandemTest), len(posLoopTest)
random.shuffle(posLoopTest)
cdg.output_loop(posLoopTest[:len(nonloopsTandemTest)] + nonloopsTandemTest, ref_genome_file, segment_size, 
                              output_file_nonloop_data, output_file_nonloop_label)



output_file_nonloop_data = 'data_nonloop_%dk%s.mat' % (segment_size/1000, 'negDiver')
output_file_nonloop_label = 'label_nonloop_%dk%s.mat' % (segment_size/1000, 'negDiver')

len(nonloopsDiverTest), len(posLoopTest)
cdg.output_loop(posLoopTest[: len(nonloopsDiverTest)] + nonloopsDiverTest, ref_genome_file, segment_size, 
                              output_file_nonloop_data, output_file_nonloop_label)




output_file_nonloop_data = 'data_nonloop_%dk%s.mat' % (segment_size/1000, 'negOneFakeWithCTCF')
output_file_nonloop_label = 'label_nonloop_%dk%s.mat' % (segment_size/1000, 'negOneFakeWithCTCF')

len(nonloopsOneFakeCTCFTest), len(posLoopTest)
cdg.output_loop(posLoopTest[: len(nonloopsOneFakeCTCFTest)] + nonloopsOneFakeCTCFTest, ref_genome_file, segment_size, 
                              output_file_nonloop_data, output_file_nonloop_label)


output_file_nonloop_data = 'data_nonloop_%dk%s.mat' % (segment_size/1000, 'negOneFakeNoCTCF')
output_file_nonloop_label = 'label_nonloop_%dk%s.mat' % (segment_size/1000, 'negOneFakeNoCTCF')

len(nonloopsOneFakeNoCTCFTest), len(posLoopTest)
cdg.output_loop(posLoopTest[: len(nonloopsOneFakeNoCTCFTest)] + nonloopsOneFakeNoCTCFTest, ref_genome_file, segment_size, 
                              output_file_nonloop_data, output_file_nonloop_label)




'''
Orientation data
'''
#######Orientation data
output_file_train_data = 'data_boundary_direction_%dk_train%s.mat' % (segment_size/1000, version)
output_file_train_label = 'label_boundary_direction_%dk_train%s.mat' % (segment_size/1000, version)

output_file_val_data = 'data_boundary_direction_%dk_val%s.mat' % (segment_size/1000, version)
output_file_val_label = 'label_boundary_direction_%dk_val%s.mat' % (segment_size/1000, version)

output_file_test_data = 'data_boundary_direction_%dk_test%s.mat' % (segment_size/1000, version)
output_file_test_label = 'label_boundary_direction_%dk_test%s.mat' % (segment_size/1000, version)



def getLeftRightBoundary(posLoop):
    left_boundary = [x.b1 for x in posLoop]
    right_boundary = [x.b2 for x in posLoop]
    
    
    common = set(left_boundary).intersection(set(right_boundary))
    len(common) #912
    
    
    
    left_boundary = [x for x in left_boundary if not x in common] #list(set(left_boundary).difference(common))
    right_boundary = [x for x in right_boundary if not x in common] #list(set(right_boundary).difference(common))
    
    #to remove duplicate boundaries
    left_boundary_dict = {x.chrom + '_' + str(x.start) + '_' + str(x.end) : x for x in left_boundary}
    right_boundary_dict = {x.chrom + '_' + str(x.start) + '_' + str(x.end) : x for x in right_boundary}
    
    left_boundary = list(left_boundary_dict.values())
    right_boundary = list(right_boundary_dict.values())
    
    len(left_boundary) #9217
    len(right_boundary) #9083
    
    
    for x in left_boundary:
        x.label = 0
    
    for x in right_boundary:
        x.label = 1
    
    
    random.shuffle(left_boundary)
    random.shuffle(right_boundary)
    
    pos_train = [x for x in left_boundary if not x.chrom in (val_chrom + test_chrom)]
    pos_val = [x for x in left_boundary if x.chrom in val_chrom]
    pos_test = [x for x in left_boundary if x.chrom in test_chrom]
    
    
    neg_train = [x for x in right_boundary if not x.chrom in (val_chrom + test_chrom)]
    neg_val = [x for x in right_boundary if x.chrom in val_chrom]
    neg_test = [x for x in right_boundary if x.chrom in test_chrom]

    
    return(pos_train + neg_train, pos_val + neg_val, pos_test + neg_test)


trainData, valData, testData = getLeftRightBoundary(posLoop)

cdg.output(trainData, ref_genome_file, segment_size, 
                              output_file_train_data, output_file_train_label, isreverse=False)

cdg.output(valData, ref_genome_file, segment_size, 
                              output_file_val_data, output_file_val_label, isreverse=False)

cdg.output(testData, ref_genome_file, segment_size, 
                              output_file_test_data, output_file_test_label, isreverse=False)








'''Check how many coiled loops in testset '''

allCTCFs = cdg.readFIMOCTCF(ctcfFile)

#boundaries with multiple CTCF motifs
for x in testData:
    x.ctcf_dir = 0
testData = cdg.addCTCFOrientation(testData, allCTCFs)


countMult = 0
incorrect = 0
for x in testData:
    if x.ctcf_dir == 3:
        countMult += 1
    if (x.label == 0 and x.ctcf_dir == 1) or (x.label == 1 and x.ctcf_dir == 2):
        incorrect += 1
        

print(countMult * 100.0 / len(testData))
print(incorrect * 100.0 / len(testData))



#coiled loops and loops with boundaries with multiple CTCFs
coilLoop = []
posLoopID = []
countTotal = 0
countTandem = 0
countMult = 0
countConv = 0
for x in posLoop:
    if x.b1.chrom in test_chrom:
        
        countTotal += 1
        posLoopID.append(x.b1.chrom + "_" + str(x.b1.start) + "_" + str(x.b1.end) + "_" + str(x.b2.start)\
                            + "_" + str(x.b2.end) + "_1")
        
        if x.b1.ctcf_dir == x.b2.ctcf_dir and x.b1.ctcf_dir != 3:
            countTandem += 1
            coilLoop.append(x.b1.chrom + "_" + str(x.b1.start) + "_" + str(x.b1.end) + "_" + str(x.b2.start)\
                            + "_" + str(x.b2.end) + "_1")
        
        if x.b1.ctcf_dir == 3 or x.b2.ctcf_dir == 3:
            countMult += 1
            
        if x.b1.ctcf_dir == 2 and x.b2.ctcf_dir == 1:
            countConv += 1

print(countTotal, countTandem, countMult)
print(countTandem * 100.0/countTotal, 100.0 * countMult/countTotal, 100.0 * countConv / countTotal)

'''check coiled loops'''
import re
posLoopTest[:len(nonloopsTandemTest)] 

nonloopsTandemTest

probFile = '/Users/tat2016/Box Sync/Research/Code/ChromatinStatePrediction/result/checkCoilLoop_output/data_nonloop_4knegTandem.txt'
prob = {}
with open(probFile, 'r') as fin:
    for ln in fin.readlines():
        st = re.split('[:\s\t]+',ln)
        prob[st[0]] = float(st[1])


coilLoopProb = []
for x in coilLoop:
    if x in prob:
        #print(x, prob[x])
        coilLoopProb.append(prob[x])

negLoopProb = []
posLoopProb = []
for x in prob:
    if not x in posLoopID:
        negLoopProb.append(prob[x])
        #print(prob[x])
    
    if x in posLoopID:
        posLoopProb.append(prob[x])
    

        
count = 0
for i in coilLoopProb:
    
    print(i, '\t',sum(i > np.array(negLoopProb)) * 100/ len(negLoopProb))
    if sum(i > np.array(negLoopProb)) * 100/ len(negLoopProb) > 80.0:
        count += 1

print(count, len(coilLoopProb)) # 15 32


count = 0 
for i in posLoopProb:
    #print(i, '\t',sum(i > np.array(negLoopProb)) * 100/ len(negLoopProb))
    if sum(i > np.array(negLoopProb)) * 100/ len(negLoopProb) > 99.0:
        count += 1    

print(count, len(posLoopProb)) 

''' '''



    
    

#variants = variant_class.extract_variants(variant_file, sample_name, all_variant = True)
#import imp
#imp.reload(cdg)
#[norm_looplist, norm_nonlooplist] = cdg.get_loop(gm12878_loop_file, variants, segment_size, 
#                                        ctcfFile = ctcfFile, nonloopType = 1, isNonLoop=False)
#
#len(norm_looplist) # 16089
#len(norm_nonlooplist) # 16050
#
#allCTCFs = cdg.readFIMOCTCF(ctcfFile)
#print('number of ctcfs: %d' % (len(allCTCFs)))
#
##looplist = boundary_class.read_loop(gm12878_loop_file)
#imp.reload(cdg)
#
#allboundaries = [x.b1 for x in norm_looplist] + [x.b2 for x in norm_looplist] + [x.b1 for x in norm_nonlooplist] + [x.b2 for x in norm_nonlooplist]
#
#for x in allboundaries:
#    x.ctcf_dir == 0
#    
#allboundaries = cdg.addCTCFOrientation(allboundaries, allCTCFs)
#
#
#print(len([x for x in norm_looplist if x.b1.ctcf_dir in [2,3] and x.b2.ctcf_dir in [1,3]]), len(norm_looplist))
#print(len([x for x in norm_looplist if x.b1.ctcf_dir in [1,3] and x.b2.ctcf_dir in [2,3]]), len(norm_looplist))
#print(len([x for x in norm_looplist if x.b1.ctcf_dir in [1] and x.b2.ctcf_dir in [1]]), len(norm_looplist))
#print(len([x for x in norm_looplist if x.b1.ctcf_dir in [2] and x.b2.ctcf_dir in [2]]), len(norm_looplist))
#
#print(len([x for x in norm_nonlooplist if x.b1.ctcf_dir in [2] and x.b2.ctcf_dir in [2]]), len(norm_nonlooplist))
#
#
#'''
#make non-loops from non-boundaries as well
#'''
#blist = boundary_class.read_boundary(gm12878_loop_file)
#blist = boundary_class.merge_boundary(blist)
#nblist = cdg.generate_negative_samples(blist, segment_size)
#
#random.shuffle(nblist)
#
#selected_nblist = nblist[:len(norm_looplist)]
#
##make non-loops from random non-boundaries
#
#random_nonloops = []
#for i in range(len(selected_nblist)-1):
#    b1 = Region(selected_nblist[i].chrom, selected_nblist[i].start, selected_nblist[i].end)
#    b2 = Region(selected_nblist[i + 1].chrom, selected_nblist[i + 1].start, selected_nblist[i + 1].end)
#    loop = Loop(b1,b2)
#    loop.label = 0
#    random_nonloops.append(loop)
#
#variant_class.overlap_variants([x.b1 for x in random_nonloops] + [x.b2 for x in random_nonloops], variants)
#    
##making half non-loops from one true boundary and one non-boundary
#randon_halfnonloops = []
#for i in range(len(selected_nblist)):
#    loop = random.choice(norm_looplist)
#    b1 = Region(selected_nblist[i].chrom, selected_nblist[i].start, selected_nblist[i].end)
#    
#    if random.choice([0,1]) == 0:
#        b2 = Region(loop.b1.chrom, loop.b1.start, loop.b1.end)
#    else:
#        b2 = Region(loop.b2.chrom, loop.b2.start, loop.b2.end)
#    
#    
#    if random.choice([0,1]) == 0:
#        tmp_loop = Loop(b1,b2)
#        
#    else:
#        tmp_loop = Loop(b2,b1)
#        
#    tmp_loop.label = 0
#    randon_halfnonloops.append(tmp_loop)
#    
#
#variant_class.overlap_variants([x.b1 for x in randon_halfnonloops] + [x.b2 for x in randon_halfnonloops], variants)
#
#
#
##for x in random_nonloops:
##    x.label = 0
##
##for x in randon_halfnonloops:
##    x.label = 0
#
#'''
#Negative loops with pairs of boundaries with convergent orientation
#'''
#import re
#
#gm12878_CTCF_file = '/Users/tat2016/Box Sync/Research/Data/ChIP_Seq/GM12878_CTCF_orientation.bed'
#
#ctcfs = []
#with open(gm12878_CTCF_file, 'r') as fin:
#    for ln in fin.readlines():
#        st = re.split('[\s\t]+', ln)
#        chrom = st[0]
#        start = int(st[1])
#        end = int(st[2])
#        direction = 0
#        if st[3] == 'Reverse':
#            direction = 1
#        elif st[3] == 'Forward':
#            direction = 2
#        
#        reg = Region(chrom, start, end)
#        reg.ctcf_dir = direction
#        
#        ctcfs.append(reg)
#        
#        
#        
##gm12878_CTCF_file = '/Users/tat2016/Box Sync/Research/Data/ctcf_positions.tsv'
##
##ctcfs = []
##with open(gm12878_CTCF_file, 'r') as fin:
##    for ln in fin.readlines():
##        
##        if ln.startswith('#'):
##            continue
##                         
##        st = re.split('[\s\t]+', ln)
##        if len(st) < 5 or '_' in st[2] or not re.search(r'^chr[0-9]+', st[2]):
##            continue
##        
##        chrom = st[2]
##        start = int(st[3])
##        end = int(st[4])
##        
##         
##        direction = 0
##        if st[5] == '-':
##            direction = 1
##        elif st[5] == '+':
##            direction = 2
##        
##        reg = Region(chrom, start, end)
##        reg.ctcf_dir = direction
##        
##        ctcfs.append(reg)
#       
#print('number of ctcf: %d' % (len(ctcfs)))       
#
##all boundaries including fake ones
#allboundaries = [x.b1 for x in norm_looplist] + [x.b2 for x in norm_looplist] + [x.b1 for x in norm_nonlooplist] + [x.b2 for x in norm_nonlooplist]
#
#for x in allboundaries:
#    x.ctcf_dir = 0
#
#allboundaries = cdg.addCTCFOrientation(allboundaries, ctcfs)
#
##refSeqFile = "/Users/tat2016/data/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa"
##def getChromlist(refSeqFile):
##    
##    #chrom list
##    chromList = {}
##    
##    for seqRecord in SeqIO.parse(refSeqFile, "fasta"):
##        #only consider chromosome 1,2,... and X and Y
##        if seqRecord.name.startswith("GL") or seqRecord.name.startswith("MT"):
##            continue
##        
##        chrom = 'chr' + seqRecord.name
##        chromList[chrom] = seqRecord
##    
##    return(chromList)        
##
##chromList = getChromlist(refSeqFile)
##len(chromList['1'])
##for x in allboundaries:
##    if x.end > len(chromList[x.chrom]):
##        print(str(x))
##
#
#for x in allboundaries:
#    x.ctcf_dir = 0
#
#allboundaries = sorted(allboundaries, key = lambda x: (x.chrid, x.start))
#ctcfs = sorted(ctcfs, key = lambda x: (x.chrid, x.start))
#
#        
#
#
#'''''''''''''''''
#'''''''''''''''''
#
#####################################
#
###generate multiple dataset with same positive samples and validation and test but different negative samples
##length with half of data
#random.shuffle(norm_nonlooplist)
#
#random.shuffle(randon_halfnonloops)
#random.shuffle(random_nonloops)
#
#norm_nonlooplist_train, norm_nonlooplist_val, norm_nonlooplist_test = [],[],[]
#for x in norm_nonlooplist:
#    if x.b1.chrom in val_chrom:
#        norm_nonlooplist_val.append(x)
#    elif x.b1.chrom in test_chrom:
#        norm_nonlooplist_test.append(x)
#    else:
#        norm_nonlooplist_train.append(x)
#        
#
#randon_halfnonloops_train, randon_halfnonloops_val, randon_halfnonloops_test = [],[],[]
#for x in randon_halfnonloops:
#    if x.b1.chrom in val_chrom:
#        randon_halfnonloops_val.append(x)
#    elif x.b1.chrom in test_chrom:
#        randon_halfnonloops_test.append(x)
#    else:
#        randon_halfnonloops_train.append(x)
#
#random_nonloops_train, random_nonloops_val, random_nonloops_test = [],[],[]
#for x in random_nonloops:
#    if x.b1.chrom in val_chrom:
#        random_nonloops_val.append(x)
#    elif x.b1.chrom in test_chrom:
#        random_nonloops_test.append(x)
#    else:
#        random_nonloops_train.append(x)
#
#        
#
#
#random.shuffle(norm_looplist)
##random.shuffle(mix_norm_nonlooplist)
#
#pos_train = [x for x in norm_looplist if not x.b1.chrom in (val_chrom + test_chrom)] 
#pos_val = [x for x in norm_looplist if x.b1.chrom in val_chrom] 
#pos_test = [x for x in norm_looplist if x.b1.chrom in test_chrom] 
#
#
##neg_train = norm_nonlooplist_train[: int(0.8 * len(pos_train))] + randon_halfnonloops_train[: int(0.1 * len(pos_train))] + random_nonloops_train[:int(0.1 * len(pos_train))]
##neg_val = norm_nonlooplist_val[: int(0.8 * len(pos_val))] + randon_halfnonloops_val[: int(0.1 * len(pos_val))] + random_nonloops_val[: int(0.1 * len(pos_val))]
##neg_test = norm_nonlooplist_test[: int(0.8 * len(pos_test))] + randon_halfnonloops_test[: int(0.1 * len(pos_test))] + random_nonloops_test[: int(0.1 * len(pos_test))]
#
##test using only fake loops with one fake boundary to see if it is enough
#neg_train = norm_nonlooplist_train[: int(0.8 * len(pos_train))] + randon_halfnonloops_train[: int(0.1 * len(pos_train))] + random_nonloops_train[:int(0.1 * len(pos_train))]
#neg_val = norm_nonlooplist_val[: int(0.8 * len(pos_val))] + randon_halfnonloops_val[: int(0.1 * len(pos_val))] + random_nonloops_val[: int(0.1 * len(pos_val))]
#neg_test = norm_nonlooplist_test[: int(0.8 * len(pos_test))] + randon_halfnonloops_test[: int(0.1 * len(pos_test))] + random_nonloops_test[: int(0.1 * len(pos_test))]
#
#
#print(len([x for x in norm_looplist if x.b1.ctcf_dir in [2,3] and x.b2.ctcf_dir in [1,3]]), len(norm_looplist))
#print(len([x for x in norm_looplist if x.b1.ctcf_dir in [1,3] and x.b2.ctcf_dir in [2,3]]), len(norm_looplist))
#print(len([x for x in norm_looplist if x.b1.ctcf_dir in [1] and x.b2.ctcf_dir in [1]]), len(norm_looplist))
#print(len([x for x in norm_looplist if x.b1.ctcf_dir in [2] and x.b2.ctcf_dir in [2]]), len(norm_looplist))
#print(len([x for x in norm_looplist if x.b1.ctcf_dir == 0 or x.b2.ctcf_dir == 0]))
#print(set([x.b2.ctcf_dir for x in norm_looplist]))
#
#print(len([x for x in neg_train if x.b1.ctcf_dir in [2,3] and x.b2.ctcf_dir in [1,3]]), len(neg_train))
#print(len([x for x in neg_test if x.b1.ctcf_dir in [2,3] and x.b2.ctcf_dir in [1,3]]), len(neg_test))
#
#
#
#
#
#
#print(set([x.b1.chrom for x in neg_val]))
#print(set([x.b1.chrom for x in neg_test]))
#print(set([x.b1.chrom for x in neg_train]))
#
#
#cdg.output_loop(pos_train + neg_train, ref_genome_file, segment_size, 
#                              output_file_train_data +'80', output_file_train_label+'80')
#
#
#cdg.output_loop(pos_val + neg_val, ref_genome_file, segment_size, 
#                              output_file_val_data+'80', output_file_val_label+'80')
#
#cdg.output_loop(pos_test + neg_test, ref_genome_file, segment_size, 
#                              output_file_test_data+'80', output_file_test_label+'80')
#
#
#
##negative samples with convergent CTCF
#
#neg_test_convCTCF = [x for x in neg_test if x.b1.ctcf_dir in [2,3] and x.b2.ctcf_dir in [1,3]]
#output_file_test_data = 'data_loop_%dk_test%s.mat' % (segment_size/1000, '_negConvCTCF')
#output_file_test_label = 'label_loop_%dk_test%s.mat' % (segment_size/1000, '_negConvCTCF')
#
#print(len(neg_test_convCTCF), len(neg_test)) #107
#
#cdg.output_loop(pos_test[:len(neg_test_convCTCF)] + neg_test_convCTCF, ref_genome_file, segment_size, 
#                              output_file_test_data, output_file_test_label)
#
#
###########
#''' Positive samples without convergent CTCF'''
#pos_test_tandem = [x for x in pos_test if x.b1.ctcf_dir == x.b2.ctcf_dir and x.b1.ctcf_dir in [1,2]]
#print(len(pos_test_tandem), len(pos_test)) # 141
#
#
#output_file_test_data = 'data_loop_%dk_test%s.mat' % (segment_size/1000, '_posTandemCTCF')
#output_file_test_label = 'label_loop_%dk_test%s.mat' % (segment_size/1000, '_posTandemCTCF')
#
#cdg.output_loop(pos_test_tandem + neg_test[: len(pos_test_tandem)], ref_genome_file, segment_size, 
#                              output_file_test_data, output_file_test_label)
#
#
##############
#
#
##common_data_generation.output_loop(random_nonloop, chrom_list, segment_size, 
##                              output_file_nonloop_data, output_file_nonloop_label)
#
#
#'''
#Orientation data
#'''
########Orientation data
#output_file_train_data = 'data_boundary_direction_%dk_train%s.mat' % (segment_size/1000, version)
#output_file_train_label = 'label_boundary_direction_%dk_train%s.mat' % (segment_size/1000, version)
#
#output_file_val_data = 'data_boundary_direction_%dk_val%s.mat' % (segment_size/1000, version)
#output_file_val_label = 'label_boundary_direction_%dk_val%s.mat' % (segment_size/1000, version)
#
#output_file_test_data = 'data_boundary_direction_%dk_test%s.mat' % (segment_size/1000, version)
#output_file_test_label = 'label_boundary_direction_%dk_test%s.mat' % (segment_size/1000, version)
#
#
#
#left_boundary = [x.b1 for x in norm_looplist]
#right_boundary = [x.b2 for x in norm_looplist]
#
#
#
#
#common = set(left_boundary).intersection(set(right_boundary))
#len(common) #912
#
#
#
#left_boundary = [x for x in left_boundary if not x in common] #list(set(left_boundary).difference(common))
#right_boundary = [x for x in right_boundary if not x in common] #list(set(right_boundary).difference(common))
#
#left_boundary_dict = {x.chrom + '_' + str(x.start) + '_' + str(x.end) : x for x in left_boundary}
#right_boundary_dict = {x.chrom + '_' + str(x.start) + '_' + str(x.end) : x for x in right_boundary}
#
#left_boundary = list(left_boundary_dict.values())
#right_boundary = list(right_boundary_dict.values())
#
#len(left_boundary) #9217
#len(right_boundary) #9083
#
#
#for x in left_boundary:
#    x.label = 0
#
#for x in right_boundary:
#    x.label = 1
#    
#
#
#len([x for x in left_boundary if len(x.variants) > 0]), len(left_boundary)
#len([x for x in right_boundary if len(x.variants) > 0]), len(right_boundary)
#
#
#
#
###generate multiple dataset with same positive samples and validation and test but different negative samples
##length with half of data
#left_len = len(left_boundary)
#
#
#random.shuffle(left_boundary)
#random.shuffle(right_boundary)
#
#pos_train = [x for x in left_boundary if not x.chrom in (val_chrom + test_chrom)]
#pos_val = [x for x in left_boundary if x.chrom in val_chrom]
#pos_test = [x for x in left_boundary if x.chrom in test_chrom]
#
#
#neg_train = [x for x in right_boundary if not x.chrom in (val_chrom + test_chrom)]
#neg_val = [x for x in right_boundary if x.chrom in val_chrom]
#neg_test = [x for x in right_boundary if x.chrom in test_chrom]
#
#
#
#
#cdg.output(pos_train + neg_train, ref_genome_file, segment_size, 
#                              output_file_train_data, output_file_train_label, isreverse=False)
#
#
#cdg.output(pos_val + neg_val, ref_genome_file, segment_size, 
#                              output_file_val_data, output_file_val_label, isreverse=False)
#
#cdg.output(pos_test + neg_test, ref_genome_file, segment_size, 
#                              output_file_test_data, output_file_test_label, isreverse=False)
#



        
    
