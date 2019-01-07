#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:04:20 2018

@author: tat2016
"""

import boundary_class
import region_class
import h5py
import variant_class
import numpy as np
from Bio.Seq import MutableSeq
import pandas as pd
import re
import random
from Bio import SeqIO
import copy

def get_overlap(x, y):
    '''Return overlap length of 2 intervals'''
    if x.chrom == y.chrom:
        return max(0, min(x.end, y.end) - max(x.start, y.start))
    else:
        return 0
    

def isAhead(x, y):
    '''Check if x is ahead of y, e.g. x.chrom > y.chrom or (x.chrom == y.chrom and x.start > y.end)'''
    
    return x.chrid > y.chrid or (x.chrid == y.chrid and x.start > y.end)
    

def remove_duplicate(regions):
    '''
    remove duplicate regions
    '''
    used = []
    tmp = []
    for x in regions:
        st = x.chrom + "_" + str(x.start) + "_" + str(x.end)
        if st not in used:
            tmp.append(x)
            used.append(st)
    return tmp
    
#    regions = sorted(regions, key = lambda x: (x.chrid, x.start))
#    rs = [regions[0]]
#    for i in range(1, len(regions)):
#        if regions[i] != rs[-1]:
#            rs.append(regions[i])
#    
#    return(rs)


def isduplicate(regions):
    '''
    check if there is any duplicate region
    '''
    return len(regions) > len(set([x.chrom + "." + str(x.start) + "." + str(x.end) for x in regions]))


def generate_negative_samples(blist, segment_size = 1000):
    '''
    This function generate negative samples from list of positive samples
    blist: positive samples
    segment_size: length of a sample (number of bases)

    
    '''
    nblist = [] #non-boundary list
    for i in range(1, len(blist)):
        if blist[i - 1].chrom != blist[i].chrom:
            continue
        
        start = blist[i - 1].end + 1
        end = blist[i].start - 1
        
        if end - start < segment_size:
            continue
    
        for k in range(int((end - start)/segment_size)):
            
            nb_start = start + k * segment_size
            #np.random.randint(low=start, high = end - segment_size)
            nb_end = nb_start + segment_size
    
            nblist.append(boundary_class.Boundary(blist[i].chrom, nb_start, nb_end))
    
    return(nblist)
    
    

def norm_aboundary(x, half_segment):
    '''
    Normalize a boundary to have length of 2 * half_segment
    '''
    center = (x.end + x.start)/2
    newstart = max(0, center - half_segment)
    newend = center + half_segment
    return region_class.Region(x.chrom, newstart, newend)

        
def len_norm(blist, segment_size):
    '''
    Normalize boundaries to have length of segment_size
    '''
    norm_blist = []
    half_segment = int(segment_size/2)
    for x in blist:
        norm_blist.append(norm_aboundary(x, half_segment))
        
    return(norm_blist)


def norm_loopboundary(looplist, segment_size):
    '''
    Normalize loop to have boundaries of segment_size
    '''
    half_segment = int(segment_size/2)

    for x in looplist:
        x.b1 = norm_aboundary(x.b1, half_segment) 
        x.b2 = norm_aboundary(x.b2, half_segment)
        
    return(looplist)

def suboutput(hdf5_file, name, seq1, seq2):
    """Output seq1 and seq2 to hdf5 file"""

    seq1_code = variant_class.lineToTensor(str(seq1).upper())
    seq2_code = variant_class.lineToTensor(str(seq2).upper())
    
    #print('length1: ', seq1_code.shape, seq2_code.shape)
     
    seq = np.hstack((seq1_code, seq2_code))    
    
    #b.seq = seq
    hdf5_file.create_dataset(name, data=seq,
                              compression="gzip", compression_opts=5)


def get_chromlist(ref_genome_file):
    """Get chromosomes in SeqIO record"""
    #chrom list
    chrom_list = {}
    
    for seq_record in SeqIO.parse(ref_genome_file, "fasta"):
        #only consider chromosome 1,2,... and X and Y
        if seq_record.name.startswith("GL") or seq_record.name.startswith("MT"):
            continue
        
        chrom = "chr" + seq_record.name
        chrom_list[chrom] = seq_record
    
    return(chrom_list)        
    
def output(data, ref_genome_file, segment_size, data_file, label_file, isreverse=True):
    '''
    output boundary data to files
    isreverse: output sequence of the reverse strand as well

    '''
    chrom_list = get_chromlist(ref_genome_file)
    
    hdf5_file = None
    hdf5_label_file = None
    
    try:
        hdf5_file = h5py.File(data_file, mode='w')
        
        if label_file:
            hdf5_label_file = h5py.File(label_file, mode='w')
        
        for b in data:
            #print(b.chrom, b.start, b.end)
            
            [seq1,seq2] = variant_class.infer_true_seq(b, chrom_list[b.chrom], segment_size)
            
            #print('length1: ', len(seq1), len(seq2))
            
            suboutput(hdf5_file, b.chrom + "_" + str(b.start) + "_" + str(b.end) + "_1" , seq1, seq2)

            #complement sequences
            if isreverse:
                if type(seq1) is MutableSeq:
                    seq3 = seq1.toseq().reverse_complement() 
                else:
                    #print(type(seq1))
                    seq3 = seq1.reverse_complement() 
                
                if type(seq2) is MutableSeq:
                    seq4 = seq2.toseq().reverse_complement()
                else:
                    #print(type(seq2))
                    seq4 = seq2.reverse_complement()
                

                if len(seq3) != segment_size or len(seq4) != segment_size:
                    raise Exception('length of sequences is wrong: %d, %d' % (len(seq3), len(seq4)))
                
                
                suboutput(hdf5_file, b.chrom + "_" + str(b.start) + "_" + str(b.end) + "_2" , seq3, seq4)


            if hdf5_label_file:
                hdf5_label_file.create_dataset(b.chrom + "_" + str(b.start) + "_" + str(b.end) + "_1", data=b.label)

                if isreverse:
                    hdf5_label_file.create_dataset(b.chrom + "_" + str(b.start) + "_" + str(b.end) + "_2", data=b.label)   

    finally:
        if hdf5_file is not None:
            hdf5_file.close()
            
        if hdf5_label_file is not None:
            hdf5_label_file.close()

def output_loop(data, ref_genome_file, segment_size, data_file, label_file):
    '''
    output loop data (with 2 boundaries) to files
    common_data_file: many loops share boundaries so common_data_file contain non-duplicate boundaries data
    data_file: contain IDs of 2 boundaries of loops, boundaries refer to common_data_file for sequence data
    lable_file: 0: no loop, 1: loop
    
    '''

    chrom_list = get_chromlist(ref_genome_file)
    
    
    hdf5_file = None
    hdf5_label_file = None
    
    try:
        hdf5_file = h5py.File(data_file, mode='w')
        if label_file:
            hdf5_label_file = h5py.File(label_file, mode='w')
        
        for loop in data:
            #print(b.chrom, b.start, b.end)
            b1, b2 = loop.b1, loop.b2
            
            [seq1,seq2] = variant_class.infer_true_seq(b1, chrom_list[b1.chrom], segment_size)
            [seq3,seq4] = variant_class.infer_true_seq(b2, chrom_list[b2.chrom], segment_size)
             
            #suboutput(hdf5_file, b1.chrom + "_" + str(b1.start) + "_" + str(b1.end) + "_1" , seq1, seq2)
            
            seq1_code = variant_class.lineToTensor(str(seq1).upper())
            seq2_code = variant_class.lineToTensor(str(seq2).upper())
            seq3_code = variant_class.lineToTensor(str(seq3).upper())
            seq4_code = variant_class.lineToTensor(str(seq4).upper())
            
            
            if len(seq1) != segment_size or len(seq2) != segment_size or\
                len(seq3) != segment_size or len(seq4) != segment_size:
                raise Exception('length of sequences is wrong: %d, %d, %d, %d' % (len(seq1), len(seq2), len(seq3), len(seq4)))
            
            seq = np.hstack((seq1_code, seq2_code, seq3_code, seq4_code)) 
            
            #b.seq = seq
            hdf5_file.create_dataset(b1.chrom + "_" + str(b1.start) + "_" + str(b1.end) + "_" + str(b2.start) + "_" + str(b2.end) + "_1", 
                                     data=seq, compression="gzip", compression_opts=5)


            #complement sequences
            
            
            seq5 = seq1.toseq().reverse_complement() if type(seq1) is MutableSeq else seq1.reverse_complement() 
            seq6 = seq2.toseq().reverse_complement() if type(seq2) is MutableSeq else seq2.reverse_complement() 
            seq7 = seq3.toseq().reverse_complement() if type(seq3) is MutableSeq else seq3.reverse_complement() 
            seq8 = seq4.toseq().reverse_complement() if type(seq4) is MutableSeq else seq4.reverse_complement() 

            if seq5 is None or seq6 is None or seq7 is None or seq8 is None:
                raise Exception('length of sequences is wrong', seq5, seq6, seq7, seq8, type(seq1))
            

            if len(seq5) != segment_size or len(seq6) != segment_size or\
                len(seq7) != segment_size or len(seq8) != segment_size:
                raise Exception('length of sequences is wrong: %d, %d, %d, %d' % (len(seq5), len(seq6), len(seq7), len(seq8)))
            
            
#            if len(seq3) != segment_size or len(seq4) != segment_size:
#                raise Exception('length of sequences is wrong: %d, %d' % (len(seq3), len(seq4)))
                
            seq5_code = variant_class.lineToTensor(str(seq5).upper())
            seq6_code = variant_class.lineToTensor(str(seq6).upper())
            seq7_code = variant_class.lineToTensor(str(seq7).upper())
            seq8_code = variant_class.lineToTensor(str(seq8).upper())
            
            
            #seq1 = np.hstack((seq5_code, seq6_code, seq7_code, seq8_code)) 
            seq1 = np.hstack((seq7_code, seq8_code, seq5_code, seq6_code))    
            
            #b.seq = seq
            hdf5_file.create_dataset(b1.chrom + "_" + str(b1.start) + "_" + str(b1.end) + "_" + str(b2.start) + "_" + str(b2.end) + "_2", 
                                     data=seq1, compression="gzip", compression_opts=5)

                
            #suboutput(hdf5_file, b.chrom + "_" + str(b.start) + "_" + str(b.end) + "_2" , seq3, seq4)

            if hdf5_label_file:
                hdf5_label_file.create_dataset(b1.chrom + "_" + str(b1.start) + "_" + str(b1.end) + "_" + str(b2.start) + "_" + str(b2.end) + "_1", 
                                               data=loop.label)
                hdf5_label_file.create_dataset(b1.chrom + "_" + str(b1.start) + "_" + str(b1.end) + "_" + str(b2.start) + "_" + str(b2.end) + "_2", 
                                               data=loop.label)   
            
    finally:
        if hdf5_file is not None:
            hdf5_file.close()
            
        if hdf5_label_file is not None:
            hdf5_label_file.close()




def read_meta_info(meta_info_file):
    """Retrieve meta information fro chromatin state data file"""
    meta_info = pd.read_csv(meta_info_file,sep="\t")

    data = meta_info.loc[meta_info['Assembly'] == 'hg19']
    
    data = data.assign(label = [re.sub('-human', '', x) for x in data['Experiment target']])
    
    #data.columns.values
    
    data = data[['File accession','label','Assembly']]
    data = data.reset_index(drop=True)
     
    return(data)


def retrieve_region_from_ID(input_file, positive_sample=False):
    '''
    Recover regions from IDs in h5py file in input_file
    '''
    regions = []
    h5_file = None
    try:
        h5_file = h5py.File(input_file,'r')
        for k,v in h5_file.items():
            #print(h5_file[k].value)
            #if retrieving positive samples only, check if the sample is positive
            if positive_sample and h5_file[k].value != 1:
                continue

            #ignore reverse sequence
            if re.search("_2$", k):
                continue

            st = k.split("_")
            regions.append(region_class.Region(st[0], int(st[1]), int(st[2])))
            
    finally:
        if h5_file:
            h5_file.close()
    
    return(regions)
    



#distance between 2 boundaries
def distance(b1, b2):
    return abs((b1.end + b1.start)/2 - (b2.end + b2.start)/2)

def pairBoundary(leftIDs, rightIDs, blist, conMT, fakeCon, loopType=1):
    '''
    Make best pair between a left boundary and a right boudary following loopType
    
    leftIDs: IDs in decreasing order
    rightIDs: IDs in increasing order
    conMT: connection matrix, con[i,j] indicates that i,j form a loop
    loopType: 
    + 1: fake loops with convergent boundaries
    + 2: fake loops with tandem boundaries
    + 3: fake loops with one true boundary, one fake boundary containing a CTCF
    + 4: fake loops with one true boundary, one fake boundary without a CTCF
    + 5: fake loops with true boundaries in divergent CTCFs

    
    '''
    #import sys
    #minDist = sys.maxsize
    
    b1,b2 = None, None
    if loopType in [1,2,5]:
        
        for i in leftIDs:
            for j in rightIDs:
                if conMT[i,j] == 0 and fakeCon[i,j] == 0 and blist[i].chrid == blist[j].chrid:
                    if loopType == 1 and blist[i].ctcf_dir in [2,3] and blist[j].ctcf_dir in [1,3]:
                        b1, b2 = blist[i], blist[j]
                        return (b1, b2)
                    
                    if loopType == 2 and blist[i].ctcf_dir in [1,2] and blist[i].ctcf_dir == blist[j].ctcf_dir:
                        b1, b2 = blist[i], blist[j]
                        return (b1, b2)
                    
                    if loopType == 5 and ((blist[i].ctcf_dir == 1 and blist[j].ctcf_dir in [2,3]) 
                                        or (blist[i].ctcf_dir in [1,3] and blist[j].ctcf_dir == 2)):
                        b1, b2 = blist[i], blist[j]
                        return (b1, b2)
    
    
    return (b1,b2)
                
        
    
            
    
def get_nonloop(orglooplist, variants, noLargeSV=False, ctcfFile=None, nonloopType=3, segment_size = 4000):
    '''
    loopType: 
        + 1: fake loops with convergent boundaries
        + 2: fake loops with tandem boundaries
        + 3: fake loops with one true boundary, one fake boundary containing a CTCF
        + 4: fake loops with one true boundary, one fake boundary without a CTCF
        + 5: fake loops with b1 reverse, b2 forward
    
    Construct non-loops from a list of loops, non-loops have convergent CTCF if allCTCFs is not None
    
    ctcf_dir = 2: forward, 1: reverse, 3: both (2 motifs)
    '''
    #make new loop so that non-loop will have different instances of boundaries
    looplist = copy.deepcopy(orglooplist)


    import sys
    
    blist = boundary_class.merge_boundary([x.b1 for x in looplist] + [x.b2 for x in looplist], isexact=True)
    
    #reassign boundaries of loops to this new merged blist (same data but different instances)
    for x in looplist:
        for y in blist:
            if x.b1 == y:
                x.b1 = y
            if x.b2 == y:
                x.b2 = y
        
    
    
    allCTCFs = readFIMOCTCF(ctcfFile)

    #set of non-loops
    nonlooplist = []
#''' non-loops are formed by closest pairs of boundaries with CTCF motif orientation'''
    
    
    if nonloopType in [1,2,5]:
        
        #map boundaries to numbers
        b2id = {}
        blist = sorted(blist, key = lambda x: (x.chrid, x.start))
        
        for i,v in enumerate(blist):
            b2id[v] = i
        
        
        #matrix indicate if 2 boundaries make a loop con[i,j] = 1 for a loop with 2 boundaries i,j    
        con = np.zeros((len(blist), len(blist)), dtype=int)
        for x in looplist:
            con[b2id[x.b1], b2id[x.b2]] = 1

        
        fakeCon = np.zeros((len(blist), len(blist)), dtype=int)
        
        for i in range(len(blist)):
            for j in range(i+1, len(blist)):
                if con[i,j] == 1:
                    
                    fb1,fb2 = None, None
                    
                    dist = distance(blist[i], blist[j]) # pick one with closest distance to dist
                    mindist = sys.maxsize
                   
                            
                    b1,b2 = pairBoundary([i], range(i + 1, j), blist, con, fakeCon, nonloopType)
                    if b1 and b2 and abs(distance(b1,b2) - dist) < mindist:
                        mindist = abs(distance(b1,b2) - dist)
                        fb1,fb2 = b1, b2
                    
                    b1,b2 = pairBoundary(range(j - 1, i, -1), [j], blist, con, fakeCon, nonloopType)
                    if b1 and b2 and abs(distance(b1,b2) - dist) < mindist:
                        mindist = abs(distance(b1,b2) - dist)
                        fb1,fb2 = b1, b2
                    
                    
                    b1,b2 = pairBoundary(range(i-1, -1, -1), [j], blist, con, fakeCon, nonloopType)
                    if b1 and b2 and abs(distance(b1,b2) - dist) < mindist:
                        mindist = abs(distance(b1,b2) - dist) < mindist
                        fb1,fb2 = b1, b2
                    

                    b1,b2 = pairBoundary([i], range(j + 1, len(blist)), blist, con, fakeCon, nonloopType)
                    if b1 and b2 and abs(distance(b1,b2) - dist) < mindist:
                        mindist = abs(distance(b1,b2) - dist) < mindist
                        fb1,fb2 = b1, b2                        

                            
                    if fb1 and fb2:
                        fakeCon[b2id[fb1], b2id[fb2]] = 1
                        fakeCon[b2id[fb2], b2id[fb1]] = 1
                        nonlooplist.append(boundary_class.Loop(fb1, fb2))
                    #else:
                        #print('No fake loop for this pair (%d,%d), %d' % (i,j, nonloopType),str(blist[i]), str(blist[j]))
        
        

#fake loops with one true boundary
    elif nonloopType in [3,4]:
        
        allDistances = [(x.b2.start + x.b2.end)/2.0 - (x.b1.start + x.b1.end)/2.0 for x in looplist]
        maxDist = np.percentile(allDistances, 90)
        minDist = np.percentile(allDistances, 10)
        
        '''negative samples center around CTCF motifs but not ovarlap with blist'''
        if nonloopType == 3:
            nblist = makeNegSampleWithChIP(blist, allCTCFs, segment_size)
        elif nonloopType == 4:
            nblist = generate_negative_samples(blist, segment_size)

        
        
        nblist = addCTCFOrientation(nblist, allCTCFs)
        
        #
        chr2nblist = {}
        for x in nblist:
            if not x.chrom in chr2nblist:
                chr2nblist[x.chrom] = []
            
            chr2nblist[x.chrom].append(x)
            
        
        
        for i in range(len(blist)):
            if blist[i].ctcf_dir in [2,3]:
                b1 = blist[i]
                if nonloopType == 3: # one fake boundary with CTCF and convergent orientation
                    selected = [x for x in chr2nblist[b1.chrom] if x.ctcf_dir in [1,3] and x.start - b1.end > minDist and x.start - b1.end < maxDist]
                    
                elif nonloopType == 4: # one fake boundary without CTCF
                    selected = [x for x in chr2nblist[b1.chrom] if x.ctcf_dir == 0 and x.start - b1.end > minDist and x.start - b1.end < maxDist]
                    
                if len(selected) > 0:
                    b2 = random.sample(selected, 1)[0]
                    
                    nonlooplist.append(boundary_class.Loop(b1, b2))
    
    
    norm_nonlooplist = norm_loopboundary(nonlooplist, segment_size)
    for x in norm_nonlooplist:
        x.b1 = region_class.Region(x.b1.chrom, x.b1.start, x.b1.end)
        x.b2 = region_class.Region(x.b2.chrom, x.b2.start, x.b2.end)
        x.label = 0
    
    nonloop_regions = [x.b1 for x in norm_nonlooplist] + [x.b2 for x in norm_nonlooplist]
    nonloop_regions = sorted(nonloop_regions, key = lambda x: (x.chrid, x.start))
    
    addCTCFOrientation(nonloop_regions, allCTCFs)
    
    variant_class.overlap_variants(nonloop_regions, variants, noLargeSV)
    
    return(nonlooplist)
    

#get loops, ready to output to file (e.g. boundaries are normalized and overlapped with variants)
def get_loop(loop_file, variants, segment_size, isNormalize=True, selected_chroms = None, 
             isNonLoop=True, noLargeSV=False, ctcfFile = None, nonloopType = 1):
    
    '''
    isNormalize: normalize boundaries of loops or not
    selected_chroms: a set of chromosomes and only loops in these chromosomes are returned
    isNonLoop = False to not generate non-loops
    noLargeSV: if true, SVs cover the whole regions will be ignored
    '''
    
    if isinstance(loop_file, str): # if loop_file is a file name
        #retrieve original loops
        looplist = boundary_class.read_loop(loop_file)
        
     
    else: #assuming loop_file is a list of loops
        looplist = copy.deepcopy(loop_file)
        

    if selected_chroms:
        looplist = [x for x in looplist if x.b1.chrom in selected_chroms]
      

    #merge loops
    #looplist = boundary_class.merge_loop(looplist)
    
    if isNormalize: #choose to normalize boundaries and loops or not
        #set of common boundaries
        blist = boundary_class.merge_boundary([x.b1 for x in looplist] + [x.b2 for x in looplist])
    
        
        #redefine boundaries of loops by common boundaries
        for x in looplist:
            for y in blist:
                if x.b1 == y:
                    x.b1 = y
                    
                if x.b2 == y:
                    x.b2 = y
        
        looplist = boundary_class.merge_loop(looplist, isexact=True)

    
    allCTCFs = []
    if ctcfFile:
        allCTCFs = readFIMOCTCF(ctcfFile)
    
    nonlooplist = []
    if isNonLoop:
        nonlooplist = get_nonloop(looplist, variants, noLargeSV, ctcfFile, nonloopType, segment_size)
    

    ################################            
            
    #normalize boundaries of loops to have standard length
    norm_looplist = norm_loopboundary(looplist, segment_size)
    #norm_nonlooplist = norm_loopboundary(nonlooplist, segment_size)
    
    #convert boundaries to regions ? why ???
    for x in norm_looplist:
        x.b1 = region_class.Region(x.b1.chrom, x.b1.start, x.b1.end)
        x.b2 = region_class.Region(x.b2.chrom, x.b2.start, x.b2.end)
        x.label = 1
    
    
    
    ####################
    
    #intersect variants with boundaries of loops
    
    variants = sorted(variants, key = lambda x: (x.chrid, x.start))
    
    loop_regions = [x.b1 for x in norm_looplist] + [x.b2 for x in norm_looplist]
    
    loop_regions = sorted(loop_regions, key = lambda x: (x.chrid, x.start))

    '''Add CTCF orientation to loop boundaries'''
    if len(allCTCFs) > 0:
        addCTCFOrientation(loop_regions, allCTCFs)
        
    variant_class.overlap_variants(loop_regions, variants, noLargeSV)
    
    
    return([norm_looplist, nonlooplist])    


def filterRegionWithMotif(regions, motifs, removedReturn = False):
    '''Keep regions contains one of motifs'''
    
    regions = sorted(regions, key = lambda x: (x.chrid, x.start))
    motifs = sorted(motifs, key = lambda x: (x.chrid, x.start))
    
    rs = []
    removed = []
    lastTF = 0
    
    for i in range(len(regions)):
        added = False
        while lastTF < len(motifs) and (regions[i].chrid > motifs[lastTF].chrid 
                          or (regions[i].chrid == motifs[lastTF].chrid and regions[i].start > motifs[lastTF].end)):
            lastTF += 1
        
        for j in range(lastTF, len(motifs)):
            if get_overlap(regions[i], motifs[j]) == motifs[j].end - motifs[j].start:
                rs.append(regions[i])
                added = True
                break
            elif motifs[j].chrid > regions[i].chrid or (motifs[j].chrid == regions[i].chrid and motifs[j].start > regions[i].end):
                break
        
        if not added:
            removed.append(regions[i])
    
    if removedReturn:
        return(rs, removed)
    
    return(rs)
    
def makeNegSampleWithChIP(posRegions, chipRegions,segmentSize=4000):
    '''Make negative regions centered around chipRegions, make sure that they don't overlap with posRegions'''
    
    negSamples = []
    for reg in chipRegions:
        tmpReg = copy.deepcopy(reg)
        center = int((tmpReg.start + tmpReg.end)/2.0)
        tmpReg.start = center - int(segmentSize/2.0)
        tmpReg.end = center + int(segmentSize/2.0)
        
        negSamples.append(tmpReg)
    
    #negSamples = remove_duplicate(negSamples)
    
    posRegions = sorted(posRegions, key = lambda x: (x.chrid, x.start))
    negSamples = sorted(negSamples, key = lambda x: (x.chrid, x.start))
    
    rs = []
    lastj = 0
    for i in range(len(negSamples)):
        
        isOverlap = False
        while lastj < len(posRegions) and isAhead(negSamples[i], posRegions[lastj]):
            lastj += 1
            
        for j in range(lastj, len(posRegions)):
            if get_overlap(negSamples[i], posRegions[j]) > 0.25 * min(negSamples[i].end - negSamples[i].start, posRegions[j].end - posRegions[j].start) :
                isOverlap = True
                break
            
            if isAhead(posRegions[j], negSamples[i]):
                break
        
        if not isOverlap:
            rs.append(negSamples[i])
    
    return(rs)
    
    
            
def readFIMOCTCF(inputFile):
    '''Read CTCF motifs from FIMO'''
    rs = []
    chrid, startid, endid, dirid, scoreid = 2,3,4,5,6 #1,2,3,4
    
    with open(inputFile, 'r') as fin:
        for ln in fin.readlines():
            st = re.split(r'[\s\t]+', ln)
            if len(st) < 5 or '_' in st[chrid] or '.' in st[chrid] or not re.search(r'^(chr)*[0-9]', st[chrid]):
                continue
            
            chrom = st[chrid]
            if not 'chr' in chrom:
                chrom = 'chr' + chrom
                
            start = int(st[startid])
            end = int(st[endid])
            direction = 2 if st[dirid] == '+' else 1
            
            reg = region_class.Region(chrom, start, end)
            reg.ctcf_dir = direction
            reg.score = float(st[scoreid])
            
            rs.append(reg)
    
    return(rs)
    
    
def addCTCFOrientation(regions, ctcfs):
    ''' Add CTCF orientation to regions'''
    
    regions = sorted(regions, key = lambda x: (x.chrid, x.start))
    ctcfs = sorted(ctcfs, key = lambda x: (x.chrid, x.start))
    
    startTF = 0
    for i in range(len(regions)):
        
        while startTF < len(ctcfs) and (ctcfs[startTF].chrid < regions[i].chrid or\
                (ctcfs[startTF].chrid == regions[i].chrid and ctcfs[startTF].end < regions[i].start)):
            startTF += 1
        
        for j in range(startTF, len(ctcfs)):
            if get_overlap(regions[i], ctcfs[j]) == ctcfs[j].end - ctcfs[j].start:
                if regions[i].ctcf_dir == 0:
                    regions[i].ctcf_dir = ctcfs[j].ctcf_dir
                    regions[i].score = ctcfs[j].score
                    
                elif regions[i].ctcf_dir != ctcfs[j].ctcf_dir:
                    regions[i].ctcf_dir = 3
                    regions[i].score = max(regions[i].score, ctcfs[j].score)
                    break
                
            elif ctcfs[j].chrid > regions[i].chrid or (ctcfs[j].chrid == regions[i].chrid and ctcfs[j].start > regions[i].end):
                break
        
    
    return(regions)









