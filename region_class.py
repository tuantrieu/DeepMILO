#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:00:35 2018

@author: tat2016
"""
import common
import re
class Region:
    
    def __init__(self, chrom, start, end):
        self.chrom = common.make_chrom(chrom)
        self.start = int(start) #included
        self.end = int(end) #not included
        
        #self.len = self.end - self.start #actual length before feeding to CNN, maybe change, used to determine padding or cutting
        
        self.chrid = common.get_chrom_id(self.chrom)
        
        #self.ex_start = 0 #extended region to provide context
        #self.ex_end = 0 #extended region to provide context
        
        self.seq = None #sequence from start to end, in matrix form: N x 4
        
        self.chromatin_state = {} #dictionary of chromatin states
        
        self.label = 0 #0: not boundary, 1: boundary
        self.score = 0 #use for log fold change
        self.variants = []
        
        self.ctcf_dir = 0 # 0: no ctcf, 1: reverse, 2: forward, 3: both reverse and forward
      
    def __eq__(self,other):
        if self.chrom != other.chrom:
            return False
        
        return self.end == other.end and self.start == other.start
        #return min(self.end, other.end) >= max(self.start, other.start)
            
    def __hash__(self):
        return self.chrid
        
    def __str__(self):
        return self.chrom + " " + str(self.start) + " " + str(self.end)
    
    def mutate(variant):
        pass
        #return (new_seq_200, new_left_1k, new_right_1k, new_left_2k, new_right_2k, new_left_5k, new_right_5k)


#lregion: list of regions
#lanno: list of regions of 'name' e.g. methylation ...
#a region must have more than 50% of annotion region to be consider
#assuming regions are not overlap
def annotate_regions(lregion, lanno, name, use_center_only=True, atac_seq=False):
    
    lregion = sorted(lregion, key = lambda x:(x.chrid, x.start))
    lanno = sorted(lanno, key = lambda x:(x.chrid, x.start))
    
    
    i,j, leni, lenj = 0,0, len(lregion), len(lanno)
    while i < leni and j < lenj:
        while i < leni and (lregion[i].chrid < lanno[j].chrid or 
                            (lregion[i].chrid == lanno[j].chrid and lregion[i].end < lanno[j].start)):
            i += 1
        
        while i < leni and j < lenj and (lregion[i].chrid > lanno[j].chrid or 
                                         (lregion[i].chrid == lanno[j].chrid and lregion[i].start > lanno[j].end)):
            j += 1
        
        if i < leni and j < lenj and lregion[i].chrid == lanno[j].chrid:
            
            if atac_seq: #for atac-seq signal, use log fold change
                
                if min(lregion[i].end, lanno[j].end) - max(lregion[i].start, lanno[j].start) >= 1:
                    #print('found')
                    lregion[i].chromatin_state[name].append(lanno[j].score)
                    
                
            else: #otherwise, take 200 bp at center of the region and check if more than half of it overlaps with annotation regions
                  #it assumes that a region doesn't span 2 annotation regions, which is unlikely
                if use_center_only:
                    rcenter = (lregion[i].end + lregion[i].start) / 2
                    rstart = max(rcenter - 100, 0)
                    rend = min(rcenter + 100, lregion[i].end)
                    #rstart = lregion[i].start
                    #rend = lregion[i].end
                    
                    d = min(rend, lanno[j].end) - max(rstart, lanno[j].start)
                    if d >= 100:
                        lregion[i].chromatin_state[name] = 1
                    
                    
                else:
                    
                    rstart = lregion[i].start
                    rend = lregion[i].end
                    d = min(rend, lanno[j].end) - max(rstart, lanno[j].start)
                    if d >= 1:
                        lregion[i].chromatin_state[name] = 1
                    
        
        j += 1
                

def read_regions(input_file):
    rs = []
    with open(input_file,"r") as fin:
        for ln in fin.readlines():
            if ln.startswith("chr") or ln[0].isdigit():
                st = ln.split()
                
                if not re.search("^(chr)*[0-9XY]+", st[0]):
                    continue
                
                rs.append(Region(st[0],st[1],st[2]))

    return rs
  

def merge(blist):
    blist = sorted(blist, key = lambda x:(x.chrid, x.start))
    rs = []
    rs.append(blist[0])
    for i in range(len(blist)):
        if rs[-1] == blist[i]:
            rs[-1] = Region(blist[i].chrom, min(blist[i].start, rs[-1].start), max(blist[i].end, rs[-1].end))
        else:
            rs.append(blist[i])
            
    return rs        
