import re
import sys
from constant import *

class Region:
    '''
    Chrom: must be like chr1, chr2, ... chrX
    '''
    def __init__(self, chrom, start, end):
        if 'chr' not in str(chrom):
            chrom = 'chr' + str(chrom)

        self.chrom = chrom

        if not re.search('chr[0-9XY]+$', chrom):
            # invalide chromosome number
            self.chrid = -1
        else:
            chr = chrom.replace('chr','').upper()
            if chr == 'X':
                self.chrid = 23
            elif chr == 'Y':
                self.chrid = 24
            else:
                self.chrid = int(chr)

        self.start = int(start)
        self.end = int(end) #exclusive

        self.score = 0 # ctcf motif strength score

    def __eq__(self, other):
        if self.chrom != other.chrom:
            return False
        size1 = self.end - self.start
        size2 = other.end - other.start
        return self.overlap(other) >= min(size1, size2) * OVERLAP_RATE
        #return self.end == other.end and self.start == other.start

    def overlap(self, other):
        if self.chrom != other.chrom:
            return -100000000
        # strictly larger because end is exclusive
        # must be positive > 0 to be overlap
        return min(self.end, other.end) - max(self.start, other.start)

    def __hash__(self):
        return self.chrid

    def __str__(self):
        return self.chrom + " " + str(self.start) + " " + str(self.end)


'''----------------------------------------------'''
class Boundary(Region):
    def __init__(self, chrom, start, end):
        Region.__init__(self, chrom, start, end)

        self.seq = '' # string of DNA sequence of this boundary
        self.label = 0 # 0: no boundary, 1: boundary
        self.ctcf_dir = 0 # CTCF direction; 1: left, 2: right, 3: both
        self.nbr_ctcf = 0 # number of CTCF motifs

        self.variants = [] # set of variants (Variant objects) overlapping this regions
        self.suffix = '' # used to distinguish same regions in gm12878 and k562 when outputting them to file


'''----------------------------------------------'''
class PatientRegion(Region):
    def __init__(self, chrom, start, end):
        Region.__init__(self, chrom, start, end)

        self.ref_score = -100
        self.patient_scores = {}
        self.gene_list = []
        self.cpn = None # copy number


'''----------------------------------------------'''
class Loop:
    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2
        self.label = 0 #

        self.suffix = '' # used to distinguish same loops in gm12878 and k562 when outputting them to file

    def __eq__(self, other):
        if self.b1.chrom != other.b1.chrom:
            return False

        return self.b1 == other.b1 and self.b2 == other.b2


'''----------------------------------------------'''
class Variant(Region):
    def __init__(self, chrom, start, end, vt, subtype, ref, alt, gt="1|1"):
        Region.__init__(self, chrom, start, end)

        if self.start == self.end:
            self.end += 1
        self.vt = vt # variant type, possible values: snp, SV,
        self.subtype = subtype # variant subtype, possible values:
        self.ref = ref # reference
        self.alt = alt # alternative
        self.gt = gt # genotype, default=1|1 and not using

        self.af = 0 # allele frequency
        self.cn = 0 # copy number

        #print('chrom:{}, start:{}, end:{}, vt:{}, subtype:{}, ref:{}, alt:{}'.format(chrom, start, end, vt, subtype, ref,
        #                                                                             alt))
        if vt == 'indel' and subtype != 'ins' and self.end - self.start != len(ref):
            sys.stderr('Unexpected indel with len(ref) is not the same as end - start: chrom:{}, start:{}, end:{}, vt:{}, subtype:{}, ref:{}'.format(chrom, start, end, vt, subtype, ref))


    def __str__(self):
        return self.chrom + " " + str(self.start) + " " + str(self.end) \
                    + " " + self.vt + " " + str(self.subtype) + " " \
                    + str(self.gt) + " " + str(self.ref) + " " + str(self.alt)

