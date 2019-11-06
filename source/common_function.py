from constant import *
from common_object import Variant, Boundary, Loop
import vcf
import sys
import random
import re
import numpy as np
import warnings


def extract_variants(input_file, sample, vt=None, svtype=None, notgt=None, qual=30, all_variant=False):
    '''
    In pyVCF, start and end are 0-based, but POS is 1-based,
    start and end is used here, so, in variants, start and ends are 0-based
    '''

    vcf_reader = vcf.Reader(open(input_file, 'r'))
    list_variant = []
    for rc in vcf_reader:
        # before 'or' is for k562 data, set notgt == None for K562
        # after is for NA12878
        if all_variant or (
                rc.QUAL is None and notgt is None and (sample is None or sample not in rc._sample_indexes)) or \
                (rc.QUAL > qual and rc.genotype(sample)['GT'] != notgt):

            # if svtype is not available for VT has no subtype
            if all_variant or ((vt is None or vt.lower() == rc.var_type.lower()) and \
                               (svtype is None or (svtype is not None and svtype.lower() in rc.var_subtype.lower()))):

                dvt = rc.var_type
                dsvtype = rc.var_subtype
                if dsvtype == None:
                    dsvtype = '.'

                start = rc.start
                # print(rc.INFO)
                if 'END' in rc.INFO:
                    end = rc.sv_end
                else:
                    end = start + len(rc.REF)

                af = -1
                if 'AF' in rc.INFO:
                    af = rc.INFO['AF']

                gt = "."
                if sample in rc._sample_indexes:  # if vcf file contains sample info
                    gt = rc.genotype(sample)['GT']
                    if gt == '0|0':
                        print('no mutation here')
                        continue

                if gt == "." and not all_variant:
                    print(start, end, dvt, dsvtype)
                    print(str(rc))
                    break

                # print(str(start) + " " + str(end) + " " + rc.var_type + " " + str(rc.var_subtype) + " " + str(rc.INFO))
                            #chrom, start, end, vt, subtype, ref, alt, gt = "1|1"
                var = Variant(rc.CHROM, start, end, dvt, dsvtype, rc.REF, rc.ALT, gt)

                var.af = af
                var.id = rc.ID

                list_variant.append(var)

    list_variant = sorted(list_variant, key=lambda x: (x.chrid, x.start))
    return list_variant


def normalize_region_len(reg, size=REGION_SIZE):
    if type(reg) == list:
        for x in reg:
            normalize_region_len(x)
        return

    if reg.end - reg.start == size:
        return

    mid = (reg.start + reg.end) / 2
    reg.start = max(0, int(mid - size/2))
    reg.end = int(mid + size/2)


def is_sig_overlap(reg1, reg2):
    reg_len = min(reg1.end - reg1.start, reg2.end - reg2.start)
    if reg1.overlap(reg2) >= 0.9 * reg_len:
        return True
    return False


def merge_overlap_region(regs, overlap_rate=0.5):
    '''
    Merge regions overlapped more than 50% into a new region
    :param li:
    :param overlap_rate: to determine overlap ratio for
    :return: a new list of regions
    '''
    regs = sorted(regs, key=lambda x: (x.chrid, x.start, x.end))
    i = 0

    for j in range(1, len(regs)):
        overlap_len = regs[i].overlap(regs[j])

        reg_len = min(regs[i].end - regs[i].start, regs[j].end - regs[j].start)

        maxend = max(regs[i].end, regs[j].end)
        minstart = min(regs[i].start, regs[j].start)

        # if regs[j].chrom == 'chr10' and regs[j].start == 95115793 and regs[j].end == 95119281:
        #     print(str(regs[i]), maxend - minstart, overlap_len, reg_len)
        #
        # if regs[i].chrom == 'chr10' and regs[i].start == 95115793 and regs[i].end == 95119281:
        #     print('i', str(regs[j]), maxend - minstart, overlap_len, reg_len)

        # if regs[j] not overlap with the last regs[i]
        # if overlap_len < reg_len/2: or #if not is_sig_overlap(regs[i], regs[j]):
        # don't merge if results too big region
        if overlap_len < reg_len * overlap_rate or maxend - minstart > REGION_SIZE * 1.5:
            regs[i + 1] = regs[j]
            i += 1
        else:
            # combine regs[i] and regs[j]
            regs[i].end = max(regs[i].end, regs[j].end)
            regs[i].start = min(regs[i].start, regs[j].start)

            if regs[i].ctcf_dir == -1:
                regs[i].ctcf_dir = regs[j].ctcf_dir

            elif regs[i].ctcf_dir != regs[j].ctcf_dir and regs[j].ctcf_dir != -1:
                regs[i].ctcf_dir = 2


    regs = regs[: i + 1]
    return(regs)

def test_merge_overlap_region():
    reg1 = Boundary('chr1', 10, 20)
    reg2 = Boundary('chr1', 15, 30)
    reg3 = Boundary('chr1', 28, 40)
    rl = [reg1, reg2, reg3, reg1]

    newreg1 = Boundary('chr1', 10, 30)

    merged_rl = merge_overlap_region(rl)

    assert len(merged_rl) == 2
    assert merged_rl[0] == newreg1
    #print(merged_rl[1])
    assert merged_rl[1] == reg3

    '''------------'''
    reg1 = Boundary('chr1', 10, 20)
    reg2 = Boundary('chr2', 15, 30)
    reg3 = Boundary('chr3', 28, 40)
    rl = [reg1, reg2, reg3]

    merged_rl = merge_overlap_region(rl)

    assert len(merged_rl) == 3
    assert merged_rl[0] == reg1
    assert merged_rl[1] == reg2


def overlap_variants(regions, variants, noLargeSV=False):
    '''
    assign variants into regions that overlap

    noLargeSV: if true, SVs cover the whole regions will be ignored

    Important note: a variant can be in multiple regions
    '''

    regions = sorted(regions, key=lambda x: (x.chrid, x.start, x.end))

    #remove duplicate region objects if there is any
    reglist = [regions[0]]
    for i in range(1, len(regions)):
        if regions[i] is not regions[i - 1]:
            reglist.append(regions[i])

    regions = reglist


    variants = sorted(variants, key=lambda x: (x.chrid, x.start, x.end))
    for x in regions:
        x.variants = []

    lastid = 0
    for reg in regions:
        while lastid < len(variants) and (variants[lastid].chrid < reg.chrid
                                          or (variants[lastid].chrid == reg.chrid and variants[lastid].end < reg.start)):
            lastid += 1

        for j in range(lastid, len(variants)):


            if reg.overlap(variants[j]) > 0:

                if not noLargeSV or (noLargeSV and reg.overlap(variants[j]) < reg.end - reg.start):
                    #print(reg, variants[j], reg.overlap(variants[j]), reg.end - reg.start)
                    reg.variants.append(variants[j])

            elif variants[j].chrid > reg.chrid or (variants[j].chrid == reg.chrid and variants[j].start > reg.end):
                break

    ##remove duplicate variants in regions
    # for reg in regions:
    #     if len(reg.variants) == 0:
    #         continue
    #
    #     varlist = [reg.variants[0]]
    #     reg.variants = sorted(reg.variants, key=lambda x: (x.chrid, x.start, x.end))
    #     for i in range(1, len(reg.variants)):
    #         if reg.variants[i].chrid != reg.variants[i - 1].chrid or reg.variants[i].start != reg.variants[i - 1].start\
    #                 or reg.variants[i].end != reg.variants[i - 1].end:
    #
    #             varlist.append(reg.variants[i])
    #
    #     reg.variants = varlist[:]




def test_overlap_variants():

    reg1 = Boundary('chr1', 10, 20)
    reg2 = Boundary('chr1', 15, 30)
    reg3 = Boundary('chr1', 28, 40)
    regions = [reg1, reg2, reg3]

    #chrom, start, end, vt, subtype, ref, alt
    var1 = Variant('chr1', 11, 12, 'snp','','A','T')
    var2 = Variant('chr1', 16, 17, 'snp', '', 'A', 'T')
    var3 = Variant('chr1', 30, 31, 'snp', '', 'A', 'T')
    var4 = Variant('chr1', 10, 41, 'snp', '', 'A', 'T')

    variants = [var1, var2, var3, var4]

    overlap_variants(regions, variants, noLargeSV=True)

    assert len(reg1.variants) == 2
    assert reg1.variants[0] == var1
    assert reg1.variants[1] == var2
    assert len(reg2.variants) == 1
    assert reg2.variants[0] == var2
    assert len(reg3.variants) == 1
    assert reg3.variants[0] == var3



def intersect_list(regs1,regs2, overlap_rate=0.5):
    '''
    Return regions in regs1 that overlap (>50%) with regions in regs2
    :param regs1:
    :param regs2:
    :return:
    '''
    regs1 = sorted(regs1, key=lambda x: (x.chrid, x.start, x.end))
    regs2 = sorted(regs2, key=lambda x: (x.chrid, x.start, x.end))

    lasti2 = 0
    rs = []
    for r1 in regs1:
        while lasti2 < len(regs2) and (regs2[lasti2].chrid < r1.chrid or
                                       (regs2[lasti2].chrid == r1.chrid and regs2[lasti2].end < r1.start)):
            lasti2 += 1

        for i2 in range(lasti2, len(regs2)):
            r2 = regs2[i2]
            if r1.overlap(r2) >= min(r2.end - r2.start, r1.end - r1.start) * overlap_rate:  # 50% of r1 must be overlapped to be considered
                rs.append(r1)
                break

            if r1.chrid < r2.chrid or (r1.chrid == r2.chrid and r1.end < r2.start):
                #lasti2 = i2
                break

    return rs

def test_intersect_list():

    reg0 = Boundary('chr1', 0, 10)
    reg1 = Boundary('chr1', 10, 20)
    reg2 = Boundary('chr1', 30, 40)
    reg3 = Boundary('chr1', 50, 60)

    reg4 = Boundary('chr1', 13, 18)
    reg5 = Boundary('chr1', 32, 38)
    reg6 = Boundary('chr1', 55, 65)

    regs1 = [reg0, reg1, reg2, reg3]
    regs2 = [reg4, reg5, reg6]

    rs = intersect_list(regs1, regs2)

    assert len(rs) == 3
    assert rs[0] == reg1
    assert rs[1] == reg2


def subtract_list(regs1,regs2, overlap_rate=0.9):
    '''
    overlap_rate: used to define if two regions are equal
    Return regions in regs1 that not overlap (>50%) with regions in regs2
    :param regs1:
    :param regs2:
    :return:
    '''
    regs1 = sorted(regs1, key=lambda x: (x.chrid, x.start, x.end))
    regs2 = sorted(regs2, key=lambda x: (x.chrid, x.start, x.end))

    lasti2 = 0
    rs = []
    for r1 in regs1:
        isin = False

        while lasti2 < len(regs2) and (regs2[lasti2].chrid < r1.chrid or
                                       (regs2[lasti2].chrid == r1.chrid and regs2[lasti2].end < r1.start)):
            lasti2 += 1

        for i2 in range(lasti2, len(regs2)):
            r2 = regs2[i2]
            if r1.overlap(r2) >= min(r1.end - r1.start, r2.end - r2.start) * overlap_rate:
                isin = True
                break

            if r1.chrid < r2.chrid or (r1.chrid == r2.chrid and r1.end < r2.start):
                break

        if not isin:
            rs.append(r1)

    return rs

def test_subtract_list():

    reg0 = Boundary('chr1', 0, 10)
    reg1 = Boundary('chr1', 10, 20)
    reg2 = Boundary('chr1', 30, 40)
    reg3 = Boundary('chr1', 50, 60)
    reg7 = Boundary('chr2', 50, 60)

    reg4 = Boundary('chr1', 13, 18)
    reg5 = Boundary('chr1', 32, 38)
    reg6 = Boundary('chr1', 55, 57)

    regs1 = [reg0, reg7, reg1, reg2, reg3]
    regs2 = [reg4, reg5, reg6, reg0]

    rs = subtract_list(regs1, regs2)

    assert len(rs) == 2
    assert rs[0] == reg3
    assert rs[1] == reg7
    #assert rs[2] ==



def sample_in_between_region(regions, size):
    '''
    Generate regions of 'size' in between this 'regions'
    Used to make negative samples without CTCF motif
    :param regions:
    :param size:
    :return:
    '''

    regions = sorted(regions, key=lambda x: (x.chrid, x.start, x.end))

    rs = []
    for j in range(1, len(regions)):
        lastreg = regions[j - 1]
        reg = regions[j]
        if reg.chrid == lastreg.chrid:
            start = lastreg.end + 1
            end = reg.start - 1
            dis = end - start # distance between this region and last region
            if dis > size:
                k = int(dis/size)
                for i in range(k):
                    b = Boundary(lastreg.chrom, start + size * i, start + size * (i + 1))
                    rs.append(b)

    return rs


def test_sample_in_between_region():

    reg0 = Boundary('chr1', 0, 10)
    reg1 = Boundary('chr1', 30, 40)
    reg2 = Boundary('chr1', 46, 55)
    reg3 = Boundary('chr1', 50, 60)

    rs = sample_in_between_region([reg0, reg1, reg2, reg3], 6)

    assert len(rs) == 4
    assert rs[0] == Boundary('chr1', 10, 16)
    assert rs[1] == Boundary('chr1', 16, 22)


def assign_ctcf_to_region(regions, ctcfs, reg_size=0):
    '''
    Assign ctcf direction to regions
    :param regions:
    :param ctcfs:
    :param reg_size: optional, when provided > 0, only check a region of reg_size around the mid point
    :return:
    '''
    regions = sorted(regions, key=lambda x: (x.chrid, x.start))
    ctcfs = sorted(ctcfs, key=lambda x: (x.chrid, x.start))

    buff_size = int((regions[0].end - regions[0].start - reg_size)/2) # buffer from start, end considering reg_size

    start_tf = 0
    for i in range(len(regions)):

        while start_tf < len(ctcfs) and (ctcfs[start_tf].chrid < regions[i].chrid or \
                                        (ctcfs[start_tf].chrid == regions[i].chrid and ctcfs[start_tf].end < regions[i].start)):
            start_tf += 1

        for j in range(start_tf, len(ctcfs)):
            if regions[i].overlap(ctcfs[j]) == ctcfs[j].end - ctcfs[j].start:
                if reg_size == 0 or (reg_size > 0 and min(regions[i].end - buff_size, ctcfs[j].end) -
                                     max(regions[i].start + buff_size, ctcfs[j].start) == ctcfs[j].end - ctcfs[j].start):

                    if regions[i].ctcf_dir == 0:
                        regions[i].ctcf_dir = ctcfs[j].ctcf_dir
                        regions[i].score = ctcfs[j].score

                    elif regions[i].ctcf_dir != ctcfs[j].ctcf_dir:
                        regions[i].ctcf_dir = 3
                        regions[i].score = max(regions[i].score, ctcfs[j].score)
                        break

            elif ctcfs[j].chrid > regions[i].chrid or (
                    ctcfs[j].chrid == regions[i].chrid and ctcfs[j].start > regions[i].end):
                break

        # if regions[i].ctcf_dir == 0:
        #     sys.stderr.write('no ctcf motif for this regions: {}\n'.format(str(regions[i])))

    return regions


def test_assign_ctcf_to_region():
    reg0 = Boundary('chr1', 0, 10)
    reg1 = Boundary('chr1', 30, 40)
    reg2 = Boundary('chr1', 46, 55)
    reg3 = Boundary('chr1', 50, 60)
    regs = [reg0, reg1, reg2, reg3]

    cf1 = Boundary('chr1', 2, 4)
    cf1.ctcf_dir = 1
    cf2 = Boundary('chr1', 5, 6)
    cf2.ctcf_dir = 2
    cf3 = Boundary('chr1', 33, 35)
    cf3.ctcf_dir = 1
    cf4 = Boundary('chr1', 42, 43)
    cf4.ctcf_dir = 3
    ctcfs = [cf1, cf2, cf3, cf4]

    regions = assign_ctcf_to_region(regs, ctcfs)

    assert len(regions) == 4
    assert regions[0].ctcf_dir == 3
    assert regions[1].ctcf_dir == 1
    assert regions[2].ctcf_dir == 0
    assert regions[3].ctcf_dir == 0


def load_ctcf_chip_boundary(input_file, size=REGION_SIZE, ismerge=True, isnorm=True):
    rs = []
    with open(input_file,'r') as fo:
        for ln in fo.readlines():
            st = ln.split()
            b1 = Boundary(st[0], int(st[1]), int(st[2]))
            if b1.chrid > 0:
                #normalize_region_len(b1, size)
                rs.append(b1)

    if ismerge:
        rs = merge_overlap_region(rs)

    if isnorm:
        for r in rs:
            normalize_region_len(r, size)

    return rs


def load_ctcf_motif_boundary(input_file, size=REGION_SIZE, ismerge=False, isnorm=True, p_value_thres=5e-5):
    '''
    set size = 0 to have ctcf motif fragment

    :param input_file:
    :param size:
    :return:
    '''
    rs = []
    with open(input_file,'r') as fo:
        for ln in fo.readlines():
            if ln.startswith('#'):
                continue

            st = ln.split()

            b1 = Boundary(st[2], int(st[3]), int(st[4]))
            strand = 1 if st[5] == '+' else 2
            b1.ctcf_dir = strand
            b1.score = float(st[7]) # p-value

            if b1.score > p_value_thres:
                continue

            if b1.chrid > 0:
                #if size > 0:
                #    normalize_region_len(b1, size)
                rs.append(b1)

    if ismerge:
        rs = merge_overlap_region(rs)

    if isnorm and size > 0:
        for r in rs:
            normalize_region_len(r, size)

    return rs

def load_ctcf_motif_JASPAR(input_file, size=REGION_SIZE, ismerge=False, isnorm=True, p_value_thres=5e-5):
    '''
    set size = 0 to have ctcf motif fragment

    :param input_file:
    :param size:
    :return:
    '''
    rs = []
    log10_thres = -1 * np.log10(p_value_thres) * 100

    with open(input_file,'r') as fo:
        for ln in fo.readlines():
            if ln.startswith('#'):
                continue

            st = ln.split()

            score = float(st[5])  # p-value

            if score < log10_thres:
                continue

            b1 = Boundary(st[0], int(st[1]), int(st[2]))
            strand = 1 if st[6] == '+' else 2
            b1.ctcf_dir = strand
            b1.score = score


            if b1.chrid > 0:
                #if size > 0:
                #    normalize_region_len(b1, size)
                rs.append(b1)

    if ismerge:
        rs = merge_overlap_region(rs)

    if isnorm and size > 0:
        for r in rs:
            normalize_region_len(r, size)

    return rs



def split_data(regions, test_chrom, val_chrom):
    if isinstance(regions[0], Boundary):
        train_data = [x for x in regions if x.chrom not in (test_chrom + val_chrom)]
        test_data = [x for x in regions if x.chrom in test_chrom]
        val_data = [x for x in regions if x.chrom in val_chrom]

    elif isinstance(regions[0], Loop):
        train_data = [x for x in regions if x.b1.chrom not in (test_chrom + val_chrom)]
        test_data = [x for x in regions if x.b1.chrom in test_chrom]
        val_data = [x for x in regions if x.b1.chrom in val_chrom]

    return (train_data, test_data, val_data)



def find_region(reg, regions):
    '''
    find a region in regions that = reg
    use binary search
    :param reg:
    :param regions: must be sorted
    :return:
    '''
    i, j = 0, len(regions)
    while i <= j:
        mid = int((i + j)/2)

        if regions[mid] == reg: # overlap more than 80%
            return regions[mid]

        elif regions[mid].chrid > reg.chrid or (regions[mid].chrid == reg.chrid and regions[mid].start > reg.end):
            j = mid - 1
        elif regions[mid].chrid < reg.chrid or (regions[mid].chrid == reg.chrid and regions[mid].end < reg.start):
            i = mid + 1
        elif regions[mid].chrid == reg.chrid and regions[mid].start < reg.start:
            i = mid + 1
        elif regions[mid].chrid == reg.chrid and reg.start < regions[mid].start:
            j = mid - 1

    sys.stderr.write('There is no match regions for: {}, {}, overlap:{}\n'.format(str(reg), str(regions[mid]), reg.overlap(regions[mid])))

    return None


def load_rad21_chiapet_boundary(input_file, size=REGION_SIZE, ismerge=True, isnorm=True):
    rs = []
    with open(input_file,'r') as fo:
        for ln in fo.readlines():
            if ln.startswith('#'):
                continue

            st = ln.split()

            b1 = Boundary(st[0], int(st[1]), int(st[2]))
            b2 = Boundary(st[3], int(st[4]), int(st[5]))

            if b1.chrid > 0 and b2.chrid > 0:
                #normalize_region_len(b1, size)
                #normalize_region_len(b2, size)
                rs.append(b1)
                rs.append(b2)

    if ismerge:
        rs = merge_overlap_region(rs)

    if isnorm:
        for r in rs:
            normalize_region_len(r, size)

    return rs

def load_rad21_chiapet_boundary_jurkat(input_file, size=REGION_SIZE, ismerge=True, isnorm=True):
    rs = []
    with open(input_file,'r') as fo:
        for ln in fo.readlines():
            if ln.startswith('#'):
                continue

            # import re
            # ln = 'chr1	27994103	28047959	chr1:27994103-28000625==chr1:28044471-28047959	6	0.129535548882624'
            st = re.split('[\s:\-=]+', ln)

            b1 = Boundary(st[3], int(st[4]), int(st[5]))
            b2 = Boundary(st[6], int(st[7]), int(st[8]))

            if b1.chrid > 0 and b2.chrid > 0:
                #normalize_region_len(b1, size)
                #normalize_region_len(b2, size)
                rs.append(b1)
                rs.append(b2)

    if ismerge:
        rs = merge_overlap_region(rs)

    if isnorm:
        for r in rs:
            normalize_region_len(r, size)

    return rs


def find_region_bruteforce(reg, regions):
    for x in regions:
        if reg == x:
            return x

    return None


def test_find_region():

    rad21_gm12878_loop_file = "/Users/tat2016/Box Sync/Research/Data/CHIA_PET/Heidari.GM12878.Rad21.mango.interactions.FDR0.2.mango.allCC.txt"
    rs = []
    with open(rad21_gm12878_loop_file,'r') as fo:
        for ln in fo.readlines():
            st = ln.split()

            b1 = Boundary(st[0], int(st[1]), int(st[2]))
            b2 = Boundary(st[3], int(st[4]), int(st[5]))

            if b1.chrid > 0 and b2.chrid > 0:
                normalize_region_len(b1, REGION_SIZE)
                normalize_region_len(b2, REGION_SIZE)
                loop = Loop(b1, b2)
                rs.append(loop)

    boundaries = [x.b1 for x in rs] + [x.b2 for x in rs]
    boundaries = merge_overlap_region(boundaries)

    allboundaries = [x.b1 for x in rs] + [x.b2 for x in rs]
    ntest = 1000

    for i in range(ntest):
        reg = random.sample(allboundaries, 1)[0]

        assert find_region(reg, boundaries) == find_region_bruteforce(reg, boundaries)

    b = Boundary('chr1', 1000, 5000)
    assert find_region(b, boundaries) == find_region_bruteforce(b, boundaries)



def load_rad21_chiapet_loop(input_file, size=REGION_SIZE, isnorm=False):
    rs = []
    with open(input_file,'r') as fo:
        for ln in fo.readlines():
            if ln.startswith('#'):
                continue
            st = re.split('[\s\t]+',ln)

            b1 = Boundary(st[0], int(st[1]), int(st[2]))
            b2 = Boundary(st[3], int(st[4]), int(st[5]))

            if b1.chrid > 0 and b2.chrid > 0:

                if isnorm and size > 0:
                    normalize_region_len(b1, size)
                    normalize_region_len(b2, size)

                loop = Loop(b1, b2)
                rs.append(loop)
    return rs


def normalize_loop(loops, size=REGION_SIZE, overlap_rate=OVERLAP_RATE):
    '''
    + Get a set of boundaries
    + Normalized boundaries
    + Reassign loops with new normalized boundaries:
        + if an old boundary overlaps > 50% a new boundary, replacing the old boundary with the new one
        (by construction, every old boundary will overlap (>50%) with a new boundary

    :param loops:
    :return:
    '''

    boundaries = [x.b1 for x in loops] + [x.b2 for x in loops]

    for x in boundaries:
        if x.end - x.start > size:
            warnings.warn('region length: {} longer than:{} before normalizing, potential information lost; {}'
                          .format(x.end - x.start, size, str(x)))
        normalize_region_len(x, size)

    boundaries = merge_overlap_region(boundaries, overlap_rate=overlap_rate)

    boundaries = sorted(boundaries, key=lambda x: (x.chrid, x.start, x.end))

    for x in loops:
        x.b1 = find_region(x.b1, boundaries)
        x.b2 = find_region(x.b2, boundaries)

    for x in loops:
        normalize_region_len(x.b1)
        normalize_region_len(x.b2)

    loops = sorted(loops, key=lambda x: (x.b1.chrid, x.b1.start, x.b1.end, x.b2.chrid, x.b2.start, x.b2.end))

    rs = [loops[0]]
    for i in range(1, len(loops)):
        if loops[i] != rs[-1]:
            rs.append(loops[i])

    return rs, boundaries

def make_nonloop_type123(loops, boundaries, max_dist, tp=1):
    '''

    :param loops:
    :param boundaries:
    :param max_dist:
    :param tp: non-boundary type
    :return:
    '''
    boundaries = sorted(boundaries, key=lambda x: (x.chrid, x.start))
    boundary_to_id = {}
    for k,v in enumerate(boundaries):
        boundary_to_id[v] = k

    con = np.zeros((len(boundaries), len(boundaries)))
    for x in loops:
        con[boundary_to_id[x.b1], boundary_to_id[x.b2]] = 1
        con[boundary_to_id[x.b2], boundary_to_id[x.b1]] = 1

    rs = []
    for i1 in range(len(boundaries)):
        for i2 in range(i1 + 1, len(boundaries)):
            if boundaries[i2].chrid > boundaries[i1].chrid or boundaries[i2].start - boundaries[i1].start >= max_dist:
                break

            if con[i1, i2] == 0 and boundaries[i2].chrid == boundaries[i1].chrid:
                if tp == 1 and boundaries[i1].ctcf_dir in [1,3] and boundaries[i2].ctcf_dir in [2,3]:
                    loop = Loop(boundaries[i1], boundaries[i2])
                    rs.append(loop)
                    con[i1, i2] = 1
                    con[i2, i1] = 1
                elif tp == 2 and boundaries[i1].ctcf_dir == boundaries[i2].ctcf_dir and boundaries[i2].ctcf_dir in [1,2]:
                    loop = Loop(boundaries[i1], boundaries[i2])
                    rs.append(loop)
                    con[i1, i2] = 1
                    con[i2, i1] = 1
                elif tp == 3 and boundaries[i1].ctcf_dir == 2 and boundaries[i2].ctcf_dir == 1:
                    loop = Loop(boundaries[i1], boundaries[i2])
                    rs.append(loop)
                    con[i1, i2] = 1
                    con[i2, i1] = 1

    return rs

def make_nonloop_type45(boundaries, ctcf_nonboundaries, max_dist, tp=4):
    '''

    :param loops:
    :param boundaries:
    :param max_dist:
    :param tp: non-boundary type
    :return:
    '''
    boundaries = sorted(boundaries, key=lambda x: (x.chrid, x.start))
    ctcf_nonboundaries = sorted(ctcf_nonboundaries, key=lambda x: (x.chrid, x.start))
    rs = []
    i2 = 0
    for bou in boundaries:
        if bou.ctcf_dir == 2:
            continue

        while i2 < len(ctcf_nonboundaries) and (ctcf_nonboundaries[i2].chrid < bou.chrid
                                                or (bou.chrid == ctcf_nonboundaries[i2].chrid and ctcf_nonboundaries[i2].end < bou.start)):
            i2 += 1

        for i in range(i2, len(ctcf_nonboundaries)):
            if bou.chrid < ctcf_nonboundaries[i].chrid or ctcf_nonboundaries[i].start - bou.start > max_dist:
                break

            if bou.chrid == ctcf_nonboundaries[i].chrid and ctcf_nonboundaries[i].start - bou.end > 0 and (tp == 5 or ctcf_nonboundaries[i].ctcf_dir in [2,3]):
                loop = Loop(bou, ctcf_nonboundaries[i])
                rs.append(loop)

    return rs
