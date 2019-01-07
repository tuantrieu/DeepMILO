#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:09:01 2018

@author: tat2016
"""

import vcf
import re
import numpy as np
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
from Bio.Seq import MutableSeq

import random
import common
import common_data_generation as cdg

class Variant:
    def __init__(self,sample, chrom, start, end, vt, svtype, ref, alt, gt="1|1"):
        '''
        vt: variant type
        svtype: SV type if vt = SV
        '''
        self.sample = sample
        if ("chr" in chrom):
            self.chrom = chrom
        else:
            self.chrom = "chr" + chrom
            
        self.chrid = common.get_chrom_id(self.chrom)  
            
        self.start = int(start)
        self.end = int(end)
        
        self.vt = vt
        self.svtype = svtype
        self.ref = ref
        self.alt = alt
        self.gt = gt
        
        self.id = ''
        
        self.af = -1 # allele frequency
        self.cn = -1 # copy number
        
    def __str__(self):
        return self.chrom + " " + str(self.start) + " " + str(self.end) \
                    + " " + self.vt + " " + str(self.svtype) + " " + str(self.gt)

    def __eq__(self,other):
        if self.chrom != other.chrom:
            return False
 
        return min(self.end, other.end) >= max(self.start, other.start)
            
    def __hash__(self):
        return self.chrid



def extract_variants(input_file, sample, vt=None, svtype=None, notgt=None, qual=30, all_variant = False, rate=1.0):
    '''
    In pyVCF, start and end are 0-based, but POS is 1-based, 
    start and end is used here, so, in variants, start and ends are 0-based
    '''
    
    vcf_reader = vcf.Reader(open(input_file,'r'))
    list_variant = []
    for rc in vcf_reader:
        #before 'or' is for k562 data, set notgt == None for K562
        #after is for NA12878
        if all_variant or (rc.QUAL is None and notgt is None and (sample is None or sample not in rc._sample_indexes)) or\
                (rc.QUAL > qual and rc.genotype(sample)['GT'] != notgt) :
            
            #if svtype is not available for VT has no subtype
            if all_variant or ((vt is None or vt.lower() == rc.var_type.lower()) and \
                     (svtype is None or (svtype is not None and  svtype.lower() in rc.var_subtype.lower()))):
                
                
                #only extract a proportion (rate) of the file
                if rate < 1.0:
                    if random.random() > rate:
                        continue
                        
                
                dvt = rc.var_type
                dsvtype = rc.var_subtype
                if dsvtype == None:
                    dsvtype = '.'
                
                start = rc.start 
                #print(rc.INFO)
                if 'END' in rc.INFO:
                    end = rc.sv_end
                else:
                    end = start + len(rc.REF)
                
                af = -1
                if 'AF' in rc.INFO:
                    af = rc.INFO['AF']
                
#                if end == None:
#                    end = rc.start + len(rc.REF)
                
                #gt = "1|1"
                gt = "."
                if sample in rc._sample_indexes: #if vcf file contains sample info
                    gt = rc.genotype(sample)['GT'] 
                    if gt == '0|0':
                        continue
                
                if gt == "." and not all_variant:
                    print(start, end, dvt, dsvtype)
                    print(str(rc))
                    break
                
                #print(str(start) + " " + str(end) + " " + rc.var_type + " " + str(rc.var_subtype) + " " + str(rc.INFO))
                
                var = Variant(sample, rc.CHROM, start, end, dvt, dsvtype, rc.REF, rc.ALT, gt)
                
                var.af = af
                var.id = rc.ID
                
                

                list_variant.append(var)
                
                

              
    list_variant = sorted(list_variant, key = lambda x: (x.chrid, x.start))
    return list_variant


#add variants to boundaries
#assuming variants in variant_file are sorted

def overlap_variants(regions, variants, noLargeSV = False):
    '''
    noLargeSV: if true, SVs cover the whole regions will be ignored
    
    Important note: a variant can be in multiple regions
    '''
    
    regions = sorted(regions, key = lambda x: (x.chrid, x.start))
    variants = sorted(variants, key = lambda x: (x.chrid, x.start))
    for x in regions:
        x.variants = []
    
    lastId = 0
    for reg in regions:
        while lastId < len(variants) and (variants[lastId].chrid < reg.chrid 
                          or (variants[lastId].chrid == reg.chrid and variants[lastId].end < reg.start)):
            lastId += 1
        
        for j in range(lastId, len(variants)):
            
            #if  variants[j].chrid == reg.chrid and max(reg.start, variants[j].start) <= min(reg.end, variants[j].end) :
            if cdg.get_overlap(reg, variants[j]) > 0:
                #if not noLargeSV or (noLargeSV and cdg.get_overlap(reg, variants[j]) < reg.end - reg.start):
                reg.variants.append(variants[j])
                
            if variants[j].chrid > reg.chrid or (variants[j].chrid == reg.chrid and variants[j].start > reg.end):
                break
     
        
        

def test_overlap_variants():
    from region_class import Region
    
    region1 = Region('chr1', 0, 1000)
    region2 = Region('chr1', 100, 600)
    
    var1 = Variant('', 'chr1', 1, 1, 'snv', '', '', '', '1|1')
    var2 = Variant('', 'chr1', 110, 112, 'snv', '', '', '', '1|1')
    var3 = Variant('', 'chr1', 599, 612, 'snv', '', '', '', '1|1')
    
    overlap_variants([region1, region2], [var1, var2, var3])
    
    assert(len(region1.variants) == 3)
    assert(len(region2.variants) == 2)
    
    for var in region1.variants:
        print(str(var))
    
    

#to compare if str1 and str2 are similar, more than 80% of them the same
#use to compare if calculated ref is same as ref
#str1 can be shorted than str2, incase the variant at the edge
def issimilar(str1, str2):
    if len(str1) > len(str2):
        return False
    
    if len(str1) == 0:
        return True
    
    d = 0
    for i in range(len(str1)):
        if str1[i] == str2[i]:
            d += 1
    
    return (d / len(str1) > 0.7)

 
def letterToIndex(letter, all_letters = 'ACGTN'):
    
    #n_letters = len(all_letters)
    
    return all_letters.find(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line, all_letters = 'ACGTN'):
    #ignore this letter if it is not ACGT
    line = re.sub('[^ACGTN]', '', line.upper())
    
    n_letters = len(all_letters)
    
    tensor = np.zeros((len(line), n_letters), dtype=int)
    for li, letter in enumerate(line):
                
        tensor[li][letterToIndex(letter)] = 1
        
    return tensor


#to flatten nested list in alternative  
flatten=lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]

#flatten(['C'])
       
#mutate reference by variants in bou and return seq1, seq2 
def infer_true_seq(bou, seq_record, segment_size=1000):
    '''
    Important note:
        seq_record: position is 0-based
        
        variants: position is 0-based, following pyVCF reader (although in VCF format, it is 1-based), 
        
                  (which is different from ICGC format http://docs.icgc.org/submission/guide/icgc-simple-somatic-mutation-format/#control_genotype-tumour_genotype)
                  (VCF format: https://samtools.github.io/hts-specs/VCFv4.1.pdf)
                  
                  insertion, deletion: start pos is immediately before inserted/deleted base (0-based) 
                                       and ALT contains the ref. base immediately before it, 
                                       e.g. REF = TC, ALT = TCA for A insertion
                                            REF = TC, ALT = T for C deletion
                                            
                  SNV: start pos is 0-based (from pyVCF reader)
                  
                  process by replacing REF with ALT, so make sure input complice that
                  
        
        boundary (a region class): 0-based start position, 1-based end position
    '''
    
    #mutate from variants with larger start site so that coordinate of other variants conserved
    #if same start positions, process smaller ones first
    bou.variants = sorted(bou.variants, key = lambda x: (x.chrid, -x.start, (x.end - x.start)))
    
    #seq_record1 = chrom_list[bou.chrom]
    seq1 = seq_record[bou.start: bou.end].seq.tomutable()
    seq2 = seq_record[bou.start: bou.end].seq.tomutable()
    
    seq = [seq1,seq2]

    for p in range(len(bou.variants)):
        
        var = bou.variants[p]
        
        vt = var.vt
        svtype = var.svtype

        
        ref = var.ref
        
        start = var.start
        end = var.end
        
        

        #print("before flatten", var.alt)
        alt = flatten(var.alt)
        
        #chr14 93145578 93145579
        
        #if gt is not available, make it 1 for all
        if var.gt == '.':
            gt = [1,1]
        else:
            
            
            gt = re.split('[/|]', var.gt)
            
            gt = [int(j) for j in gt]
        
        #cna = len(gt) #number of copies, assuming 2 for now
        
        #print("vt: %s, svtype: %s, ref: %s, alt: %s, gt: %s" % (vt, svtype, ref, alt, gt))
        
        offset = start - bou.start # start of the variant relative to the begining of boundary
        
        #print(seq1[offset-3],seq1[offset-2],seq1[offset-1],seq1[offset], seq1[offset+1],seq1[offset+2],seq1[offset+3])
    
        #if str(seq[0][offset: offset + len(ref)]) != ref:
        
        #ignore CNV for now because they are often larger than boundaries
        #if svtype == 'CNV':
        #    continue
        
        if offset < 0 and vt != 'sv': #variant starts before boundary.start
            prefix_ig = abs(offset) #ignore begining of ref, and alternative before prefix_ig, this does not apply to SV
            offset = 0
            #ref = ref[prefix_ig:]
            
            if len(ref) > prefix_ig:
                ref = ref[prefix_ig:]
            else:
                ref = ''    
                
            for k in range(len(alt)):
                if len(alt[k]) > prefix_ig:
                    alt[k] = str(alt[k])[prefix_ig:]
                else:
                    alt[k] = ''
            
        
        if vt != 'sv' and svtype != 'del' and not issimilar(str(seq[0][offset: offset + len(ref)]),ref):
            print("boundary, start: %d, end: %d; variant, start: %d, end: %d" % (bou.start, bou.end, start, end) )
            print("vt: %s, svtype: %s, ref: %s, alt: %s, gt: %s, offset: %d" % (vt, svtype, ref, alt, gt, offset))
            print("#: %d, variant: %s vs. cal.ref.sequence: %s vs. ref.seq: %s, before mutating: %s" 
                  % (p,ref, str(seq[0][offset: offset + len(ref)]), str(seq_record[start:end].seq),
                     str(seq_record[bou.start: bou.end][offset - 1: offset + len(ref) + 1].seq)))
            
            print('This is just a warning, it is ok if ref %s == %s before_mutating (or before_mutating[1:-1])' %\
                        (ref,str(seq_record[bou.start: bou.end][offset: offset+len(ref)].seq)))
            #it happens because two variants at the same position
            print("Mismatch-----------")
         
#        if svtype == 'ins':
#            print("%s : %s" % (ref, str(seq[0][offset: offset + len(ref)]) ))
            
        if len(gt) == 1: #data error, ignore it
            print(gt[0], start, end, var.chrom, var.gt)
            continue
            
            
        if vt == 'snp':

             for j in range(len(seq)):
                if gt[j] > 0:
                    
                    
                    
                    st = str(alt[gt[j]-1])
                    
                    if offset + len(st) >= len(seq[j]): # if del ends beyond boundary
                        seq[j] = seq[j][0:offset]
                    else:
                        seq[j] = seq[j][0:offset] + seq[j][offset + len(st) : len(seq[j])]
                    
                    seq[j] = seq[j][ :offset] + st + seq[j][offset:]
                    

#                if re.search('[^ACGTN]', str(seq[j].toseq())):
#                    r = re.findall('[^ACGTN]', str(seq[j].toseq()))
#                    for ir in r:
#                        print(ir)
#                    
#                    print('trouble variant', str(var), alt, var.alt, var.ref)                    
                
                    
                    
                
        elif vt == 'indel':
    
            if svtype == 'ins':
                
                for j in range(len(seq)):
                    if gt[j] > 0:

                        st = str(alt[gt[j]-1])
                        
                        st = re.sub('[^ACGTN]', '',st)
                        
#                        if st[0] != ref[0]:
#                            print('Input data problem, mismatch in ins (indel), ref: %s, alt: %s' 
#                                  % (ref, st))
                        
                        #first, delete REF, and then insert ALT in place of REF
                        if offset + len(ref) >= len(seq[j]): # if del ends beyond boundary
                            seq[j] = seq[j][0:offset]
                        else:
                            seq[j] = seq[j][0:offset] + seq[j][offset + len(ref) : len(seq[j])]

                        
                        #quick and dirty, insert char from the end, one-by-one
                        #acceptable because insertion is often short
                        #for k in range(len(st)): # 
                        #    seq[j].insert(offset, st[-1 - k])
                        
                        seq[j] = seq[j][ :offset] + st + seq[j][offset:]
                
                        
            elif svtype == 'del' or svtype == 'unknown':
                
                for j in range(len(seq)): #each seq is for one genotype
                    if gt[j] > 0:
                        #delete first
                        if offset + len(ref) >= len(seq[j]): # if del ends beyond boundary
                            seq[j] = seq[j][0:offset]
                        else:
                            seq[j] = seq[j][0:offset] + seq[j][offset + len(ref) : len(seq[j])]

                        
                        #insert alt if alt is available
                        if len(alt) > 0 and len(alt[gt[j] -1]) > 0:
                            st = str(alt[gt[j]-1]) #alternative correspond to alternative
                            st = re.sub('[^ACGTN]', '',st)
                            
                            
#                            if st[0] != ref[0]:
#                                print('Input data problem, mismatch in del (indel), ref: %s, alt1: %s, alt2: %s' 
#                                      % (ref, alt[0], alt[1]))
                            
                            #for k in range(len(st)):
                            #    seq[j].insert(offset, st[-1 - k])  
                            seq[j] = seq[j][ :offset] + st + seq[j][offset:]
                        
    
        elif vt == 'sv':
            if svtype == 'DEL':
                offset = max(0, offset) # DEL starts before boundary starts
                start = max(start, bou.start) # if DEL start before boundary starts
                #end = min(end, bou.end) # if DEL ends beyond boundary
                
                for j in range(len(seq)):
                    if gt[j] > 0:
                        
                        if offset + end - start >= len(seq[j]): #deletion beyond end of boundary
                            seq[j] = seq[j][0:offset]
                            
                        else: #otherwise, chop off the part from start to end, relatively to offset
                            seq[j] = seq[j][0:offset] + seq[j][offset + end - start : len(seq[j])]
                        
            elif svtype == 'INV':
                for j in range(len(seq)):
                    if gt[j] > 0:
                        if offset >= 0:
                            seq[j] = seq[j][0:offset] + seq[j][offset: offset + end - start][::-1]\
                                        + seq[j][offset + end - start:]
                            
                        else: # if variant starts before boundary starts
                            '''retrieve the sequence from the reference seq
                            'reverse it and take [: variant.end - bou.start]
                            'add it to the rest of seq[j]'''
                            seq[j] = seq_record[start:end].seq.tomutable()[::-1][bou.start - start :] + seq[j][end - bou.start:]


            elif svtype == 'DUP' or svtype == 'DUP:TANDEM':
                for j in range(len(seq)):
                    if gt[j] > 0:
                        if offset >= 0:
                            seq[j] = seq[j][0:offset + end - start] + seq[j][offset: offset + end - start] + seq[j][offset + end - start:]
                            
                        else: # if variant starts before boundary starts
                            #offset is negative, the portion not in the seq[j]
                            seq[j] = seq[j][0:offset + end - start ] + seq_record[start:end].seq.tomutable() + seq[j][offset + end - start:]
                            
                            
                
    #ignore for now because CNV often bigger than boundaries                       
            elif svtype == 'CNV':
                for j in range(len(seq)):
                    if gt[j] > 0:
                        st = str(alt[gt[j]-1])
                        cp = int(re.findall(r'\d+',st)[0]) #extract copy number
                        
                        if cp == 0: #deletion
                            offset = max(0, offset) #DEL starts before boundary starts
                            start = max(start, bou.start) # if DEL start before boundary starts
                            
                            if offset + end - start >= len(seq[j]):
                                seq[j] = seq[j][0:offset]
                            else: 
                                seq[j] = seq[j][0:offset] + seq[j][offset + end - start : len(seq[j])]
                            
                            #for k in range(end - start):
                            #    seq[j].pop(offset) #delete end - start times at the offset position
                                
                        else: #insert cp times from the end
                            #there is a case when start of variant is before start of the boundary
                            #then offset is negative ()
                            
                            if end >= bou.end: #no need to add copy number ? assuming seq[j] is at length of segment_size
                                continue
                            
                            st = seq_record[start:end].seq.tomutable()
                            
                            
                            offset_end = max(0,offset) + (end - max(bou.start, start))
                            
                            #print('test', offset_end, max(0,offset), (end - max(bou.start, start)))
                            
                            if offset_end < len(seq[j]):
                                #for _ in range(cp-1): #cp = 2, mean add one more
                                    
                                    #for k in range(len(st)):
                                        #seq[j].insert(offset_end, st[-1 - k])
                                        
                                seq[j] = seq[j][:offset_end] + (str(st) * (cp - 1)) + seq[j][offset_end :]
            

       
                                
                                    
                                    
    #truncating or padding if len(seq[j]) != segment_size
    half_segment = int(segment_size/2)
    for i in range(len(seq)):
        if len(seq[i]) < segment_size: #padding
            ad_len = segment_size - len(seq[i])
            ad_len1 = int(ad_len/2)
            ad_len2 = ad_len - ad_len1
            
            lenseq = len(seq[i])
            for k in range(ad_len2):
                seq[i].insert(lenseq,'N')
                
            for k in range(ad_len1):
                seq[i].insert(0,'N') #padding N
                
        #0 1 2 3 4 5 6
        #center = 7/2 = 3
        #k = 0
        #half = 3
        #[0 : 3] + [3: 6]
        elif len(seq[i]) > segment_size: #truncating

            center = int(len(seq[i]) / 2)
            
#            if type(seq[i]) is MutableSeq:
#                st = str(seq[i].toseq())
#            else:
#                st = str(seq[i])
#            
#            st = re.sub('[^ACGT]', '',st)
#            
#            seq[i] = Seq(st)
            
            seq[i] = seq[i][center - half_segment : center] + seq[i][center : center + half_segment]
        
        if re.search('[^ACGTN]', str(seq[i].toseq())):
            r = re.findall('[^ACGTN]', str(seq[i].toseq()))
            for ir in r:
                print(ir)
                
            print('unexpected char in string')
            
            
        if len(seq[i]) != segment_size:
            print('wrong length:', len(seq[i]))
    
    
    return(seq)


####################
def test_infer_true_seq():
    
    from region_class import Region
    from Bio import SeqIO
    
    ref_genome_file = "/Users/tat2016/data/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa"
    
    
    chrom_list = {}
    
    for seq_record in SeqIO.parse(ref_genome_file, "fasta"):
        #only consider chromosome 1,2,... and X and Y
        if seq_record.name.startswith("GL") or seq_record.name.startswith("MT"):
            continue
        
        chrom = "chr" + seq_record.name
        chrom_list[chrom] = seq_record
        if chrom == 'chr1':
            break
    
    seq_record1 = chrom_list['chr1']
    
    seq = seq_record1[20000:20010] #CCTGGTGCTC
    
    
    from variant_class import Variant
    import variant_class
    import imp
    imp.reload(variant_class)


    reg = Region('chr1',0,10)
    var = Variant('NA12878', 'chr1', 2, 2, 'snp', '', 'T', ['GC','A'], '1|2')
    
    reg.variants.append(var)
    
    trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    print(str(trueSeq[0]), '\n', str(seq))
    assert(str(trueSeq[0]) == 'CCGCGTGCTC')
    

    reg = Region('chr1',0,10)
    var = Variant('NA12878', 'chr1', 2, 3, 'snp', '', 'T', ['GC','A'], '1|2')
    
    reg.variants.append(var)
    
    trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    print(str(trueSeq[0]), '\n', str(seq))
    assert(str(trueSeq[0]) == 'CCGCGTGCTC')


    
    reg = Region('chr1',0,10)
    var = Variant('NA12878', 'chr1', 2, 3, 'indel', 'ins', 'T', ['TAC','A'], '1|2')
    
    reg.variants.append(var)
    
    trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    print(str(trueSeq[0]), '\n', str(seq))
    
    #seq = CCTACGGTGCT C
    assert(str(trueSeq[0]) == 'CTACGGTGCT')
    
 
    reg = Region('chr1',0,10)
    var = Variant('NA12878', 'chr1', 2, 2, 'indel', 'ins', 'T', ['TAC','A'], '1|2')
    
    reg.variants.append(var)
    
    trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    print(str(trueSeq[0]), '\n', str(seq))
    
    #seq = CCTACGGTGCT C
    assert(str(trueSeq[0]) == 'CTACGGTGCT')
    
    
    reg = Region('chr1',0,10)
    var = Variant('NA12878', 'chr1', 2, 3, 'indel', 'del', 'TG', 'T')
    
    reg.variants.append(var)
    trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    assert(str(trueSeq[0]) == 'CCTGTGCTCN')
    
    
    
    reg = Region('chr1',0,10)
    var = Variant('NA12878', 'chr1', 1, 5, 'sv', 'DEL', '', '')
    
    #seq = CCTGGTGCTC, NNCTGCTCNN
    reg.variants.append(var)
    trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    assert(str(trueSeq[0]) == 'NNCTGCTCNN')
    
    
    reg = Region('chr1',3,10)
    var = Variant('NA12878', 'chr1', 0, 6, 'sv', 'INV', '', '')
    
    #seq = CCTGGTGCTC, CCT GGTGCTC
    #      CCTGGT GCTC
           
    reg.variants.append(var)
    trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    assert(str(trueSeq[0]) == 'NTCCGCTCNN')
    
   
    #seq[0:5].seq.tomutable()[::-1][3:] + seq[3:10][5 - 3:]
    
    
    reg = Region('chr1',0,10)
    var = Variant('NA12878', 'chr1', 0, 3, 'sv', 'DUP', '', '')
    
    #seq = CCTGGTGCTC, 
    #      CCTCCTGGTGCTC
           
    reg.variants.append(var)
    trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    assert(str(trueSeq[0]) == 'CTCCTGGTGC')
    
    
    
    import imp
    imp.reload(variant_class)
    
    
    reg = Region('chr1',1,10)
    var = Variant('NA12878', 'chr1', 0, 3, 'sv', 'CNV', '', alt = '[3,3]')
    
    #seq = CCTGGTGCTC, C CT CCTCCTGGTG CTC
    
    #CT CCTCCTGGTG CTC
    #      CCT CCTCCTGGTG CTC
           
    reg.variants.append(var)
    trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    assert(str(trueSeq[0]) == 'CCTCCTGGTG')
    

    
