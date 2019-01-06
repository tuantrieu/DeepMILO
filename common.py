#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:58:41 2018

@author: tat2016
"""

def make_chrom(chrom):
    if "chr" in chrom:
        return chrom
    else:
        return "chr" + chrom


def get_chrom_id(chrom):
    if chrom == "chrX" or chrom == "X":
        return 23
    elif chrom == "chrY" or chrom == "Y":
        return 24
    elif chrom == "chrM" or chrom == "M":
        return 25
    else:
        return int(chrom.replace("chr",""))
