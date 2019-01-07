#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:57:01 2018

@author: tat2016
"""
import common
import re
import functools
import copy

class Boundary:
    def __init__(self, chrom, start, end):
        self.chrom = common.make_chrom(chrom)
        self.start = int(start) #included
        self.end = int(end) #not included
        
        self.chrid = common.get_chrom_id(self.chrom)
        
        self.label = -1
        
        self.ctcf_dir = 0
    
    #equal if they overlap more than 50%
    def __eq__(self,other):
        if self.chrom != other.chrom:
            return False
        
        d = min(self.end, other.end) - max(self.start, other.start)
        d1 = other.end - other.start
        d2 = self.end - self.start
        return (d * 100.0 / d1 > 50.0 or d * 100.0 / d2 > 50.0)
#        center1 = (self.start + self.end)/2.0
#        center2 = (other.start + other.end)/2.0
#        return(abs(center1 - center2) <= 5)
            
            
    def __hash__(self):
        return self.chrid
        
    def __str__(self):
        return self.chrom + " " + str(self.start) + " " + str(self.end)

class Loop:
    def __init__(self, b1, b2):
        if b1.chrid >= b2.chrid:
            self.b1 = b1
            self.b2 = b2
        else:
            self.b1 = b2
            self.b2 = b1
        
        self.label = -1
    
    def __eq__(self, other):
        
        return self.b1.chrid == other.b1.chrid and self.b1 == other.b1 and self.b2 == other.b2
    
    def __hash__(self):
        return self.b1.chrid
    
    def __str__(self):
        return self.b1.chrom + " " + str(self.b1.start) + " " + str(self.b1.end) + " "\
                + self.b2.chrom + " " + str(self.b2.start) + " " + str(self.b2.end)
        

def read_loop(input_file):
    rs = []
    with open(input_file,"r") as fin:
        for ln in fin.readlines():
            st = ln.split()
            
            if not re.search("^(chr)*[0-9XY]+", st[0]):
                continue
            b1 = Boundary(st[0],st[1],st[2])
            b2 = Boundary(st[3],st[4],st[5])
            rs.append(Loop(b1, b2))
    return rs


def compare_loop(l1, l2):
    if l1.b1.chrid != l2.b1.chrid:
        return l1.b1.chrid - l2.b1.chrid
    
    if l1.b1.start != l2.b1.start:
        return l1.b1.start - l2.b1.start
    
    if l1.b2.start != l2.b2.start:
        return l1.b2.start - l2.b2.start
    
    return 0
    

def is_exact_equal_b(b1, b2):
    return b1.chrid == b2.chrid and b1.start == b2.start and b1.end == b2.end

def is_exact_equal_l(loop1, loop2):
    return is_exact_equal_b(loop1.b1, loop2.b1) and is_exact_equal_b(loop1.b2, loop2.b2)
    
def merge_loop(llist, isexact=False):
    llist = sorted(llist, key = lambda x:(x.b1.chrid, x.b1.start, x.b2.start))
    #llist = sorted(llist, key=functools.cmp_to_key(compare_loop))
    
    rs = []
    rs.append(llist[0])
    for i in range(1, len(llist)):
        if (isexact and is_exact_equal_l(rs[-1], llist[i])) or (not isexact and rs[-1] == llist[i]):
            
            rs[-1] = Loop(merge_boundary(rs[-1].b1, llist[i].b1), merge_boundary(rs[-1].b2, llist[i].b2))
            
                    #Loop(Boundary(rs[-1].b1.chrom, min(rs[-1].b1.start, llist[i].b1.start), max(rs[-1].b1.end, llist[i].b1.end)),
                    #      Boundary(rs[-1].b2.chrom, min(rs[-1].b2.start, llist[i].b2.start), max(rs[-1].b2.end, llist[i].b2.end)))
              
        else:
            rs.append(llist[i])
            
    return rs

def test_mergeloop():
    b1 = Boundary('chr1', 0, 100)
    b2 = Boundary('chr1', 1000, 1100)
    loop1 = Loop(b1,b2)
    
    b3 = Boundary('chr1', 40,150)
    b4 = Boundary('chr1', 1040, 1150)
    loop2 = Loop(b3,b4)
    
    b5 = Boundary('chr1', 0, 150)
    b6 = Boundary('chr1', 1000, 1150)
    loop3 = Loop(b5, b6)
    
    
    
    loops = [loop1, loop2]
    
    llist = merge_loop(loops)
    
    assert( len(llist) == 1)
    assert(is_exact_equal_l(llist[0], loop3))

    
    b1 = Boundary('chr1', 0, 100)
    b2 = Boundary('chr1', 1000, 1100)
    loop1 = Loop(b1,b2)
    
    b3 = Boundary('chr1', 100,150)
    b4 = Boundary('chr1', 1040, 1150)
    loop2 = Loop(b3,b4)
    
    loops = [loop1, loop2]
    
    llist = merge_loop(loops)
    
    assert( len(llist) == 2)
    assert(is_exact_equal_l(llist[0], loop1))
    assert(is_exact_equal_l(llist[1], loop2))
    
    
    
    b1 = Boundary('chr1', 0, 100)
    b2 = Boundary('chr1', 1000, 1100)
    loop1 = Loop(b1,b2)
    
    
    loops = [loop1, copy.deepcopy(loop1), copy.deepcopy(loop1)]
    
    llist = merge_loop(loops)
    
    assert( len(llist) == 1)
    assert(is_exact_equal_l(llist[0], loop1))

    
    
    
    

def read_boundary(input_file, isboundary=True):
    rs = []
    with open(input_file,"r") as fin:
        for ln in fin.readlines():
            st = ln.split()
            
            if not re.search("^(chr)*[0-9XY]+", st[0]):
                continue
            
            rs.append(Boundary(st[0],st[1],st[2]))
            if isboundary:
                rs.append(Boundary(st[3],st[4],st[5]))
                    
    return rs

#merge duplicate boundaries
def merge_boundary(blist, isexact=False):
    '''
    isexact: True to remove exact duplicate ones
    '''
    blist = sorted(blist, key = lambda x:(x.chrid, x.start))
    rs = []
    rs.append(blist[0])
    for i in range(len(blist)):
        if (isexact and is_exact_equal_b(rs[-1], blist[i])) or (not isexact and rs[-1] == blist[i]):
            
            boundary = Boundary(blist[i].chrom, min(blist[i].start, rs[-1].start), max(blist[i].end, rs[-1].end))
            #merge ctcf orientation
            if blist[i].ctcf_dir == 0:
                boundary.ctcf_dir = rs[-1].ctcf_dir
                
            elif rs[-1].ctcf_dir == 0:
                boundary.ctcf_dir = blist[i].ctcf_dir
                
            elif rs[-1].ctcf_dir != blist[i].ctcf_dir:
                boundary.ctcf_dir = 3
            else:
                boundary.ctcf_dir = rs[-1].ctcf_dir
                
            
            rs[-1] = boundary
            
        else:
            rs.append(blist[i])
            
    return rs
    
def test_merge_boundary():
    
    b1 = Boundary('chr1', 0, 100)
    b2 = Boundary('chr1', 40,150)
    b3 = Boundary('chr1', 1000, 1100)
    
    b4 = Boundary('chr1', 0, 150)
    
    bs = merge_boundary([b1,b2,b3])
    
    print(str(bs[0]), '\n', str(b4))
    
    assert(len(bs) == 2)
    assert(is_exact_equal_b(bs[0], b4))
    assert(is_exact_equal_b(bs[1], b3))
    
    
    b1 = Boundary('chr1', 0, 100)
    b2 = Boundary('chr1', 40,150)
    b3 = Boundary('chr1', 1000, 1100)
    
    b4 = Boundary('chr1', 0, 150)
    
    bs = merge_boundary([b1,copy.deepcopy(b3),copy.deepcopy(b1),b2,b3], True)
    
    print(str(bs[0]), '\n', str(b4))
    
    assert(len(bs) == 3)
    assert(is_exact_equal_b(bs[0], b1))
    assert(is_exact_equal_b(bs[1], b2))
    assert(is_exact_equal_b(bs[2], b3))
    
    
    
    
    
    
    