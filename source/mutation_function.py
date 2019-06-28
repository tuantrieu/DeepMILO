'''
Mutate a boundary given a mutation
'''
import sys
import warnings
import re
from constant import *

flatten = lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]


def padding_truncating(seq, segment_size=REGION_SIZE):

    # truncating or padding if len(seq[j]) != segment_size
    half_segment = int(segment_size / 2)

    if len(seq) < segment_size:  # padding
        ad_len = segment_size - len(seq)
        ad_len1 = int(ad_len / 2)
        ad_len2 = ad_len - ad_len1

        lenseq = len(seq)
        for k in range(ad_len2):
            seq.insert(lenseq, 'N')

        for k in range(ad_len1):
            seq.insert(0, 'N')  # padding N

    # 0 1 2 3 4 5 6
    # center = 7/2 = 3
    # k = 0
    # half = 3
    # [0 : 3] + [3: 6]
    elif len(seq) > segment_size:  # truncating

        center = int(len(seq) / 2)

        seq = seq[center - half_segment: center] + seq[center: center + half_segment]

    if re.search('[^ACGTN]', str(seq.toseq())):
        r = re.findall('[^ACGTN]', str(seq.toseq()))
        for ir in r:
            sys.stderr.write('unexpected char: {}'.format(ir))

    if len(seq) != segment_size:
        sys.stderr.write('wrong length:', len(seq))

    return seq

def test_padding_truncating():

    from Bio.Seq import Seq
    seq = Seq('ACTCGA')
    norm_seq = padding_truncating(seq.tomutable(), segment_size=12)

    assert str(norm_seq) == 'NNNACTCGANNN'

def mutate_snp(bou, var, seq, seq_record):
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

    #seq = seq_record[bou.start: bou.end].seq.tomutable()

    if bou.overlap(var) < 0:  # == 0 is acceptable because var.start == var.end in some format
        warnings.warn('Mutation and boundary are not overlapped, boundary: {}, var: {}'.format(str(bou), str(var)))
        return seq

    vt = var.vt

    if vt != 'snp':
        sys.stderr.write('This is not snp\n')

    svtype = var.subtype

    ref = var.ref

    start = var.start
    end = var.end

    alt_list = flatten(var.alt)
    alt = ''
    if len(alt_list) == 1:
        alt = str(alt_list[0])
    elif len(alt_list) > 1:
        alt = str(alt_list[0]) if len(str(alt_list[0])) > 0 else str(alt_list[1])


    # if len(alt) == 0:
    #     sys.stderr.write('Alternative is blank, check variant: {}'.format(str(var)))

    offset = start - bou.start  # start of the variant relative to the beginning of boundary

    # if offset <= 0:
    #     warnings.warn('-----Ignore this variant:{}, {}'.format(str(var), str(bou)))

    if vt == 'snp' and offset > 0:  # can ignore if offset < 0 because segment size is pretty big already

        if seq[offset: min(offset + len(ref), len(seq))] != ref:
            warnings.warn('----------------------------')
            warnings.warn('offset:{}, len seq: {}, var: {}'.format(offset, len(seq), str(var)))
            warnings.warn('reference seq of this boundary: {} is different from ref in variant: {}'.format(seq[offset: min(offset + len(ref), len(seq))],
                                                                                          ref))

            warnings.warn('reference seq (original): {} should be the same ref in variant: {}'.format(
                str(seq_record[bou.start: bou.end][offset: offset + len(ref)].seq), ref))
            warnings.warn('----------------------------')

        # first, removing the reference seq part
        if offset + len(alt) >= len(seq):  # if snp ends beyond boundary
            seq = seq[0:offset] + alt
        else:
            #print(seq[0:offset], alt, seq[offset + len(ref): len(seq)])
            seq = seq[0:offset] + alt + seq[offset + len(ref): len(seq)]

    return seq


def test_mutate_snp():

    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Alphabet import generic_dna
    from common_object import Boundary, Variant

    seq = SeqRecord(Seq('CCTGGTGCTC', generic_dna))

    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 2, 2, 'snp', '', 'T', 'A')

    reg.variants.append(var)
    true_seq = mutate_snp(reg, var, seq)

    assert str(true_seq) == 'CCAGGTGCTC'

    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 1, 1, 'snp', '', 'C', 'AT')

    reg.variants.append(var)
    true_seq = mutate_snp(reg, var, seq)

    assert str(true_seq) == 'CATTGGTGCTC'


    # reg = Region('chr1', 0, 10)
    # var = Variant('NA12878', 'chr1', 2, 3, 'indel', 'ins', 'T', ['TAC', 'A'], '1|2')
    #
    # reg.variants.append(var)
    #
    # trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    # print(str(trueSeq[0]), '\n', str(seq))
    #
    # # seq = CCTACGGTGCT C
    # assert (str(trueSeq[0]) == 'CTACGGTGCT')
    #
    # reg = Region('chr1', 0, 10)
    # var = Variant('NA12878', 'chr1', 2, 2, 'indel', 'ins', 'T', ['TAC', 'A'], '1|2')
    #
    # reg.variants.append(var)
    #
    # trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    # print(str(trueSeq[0]), '\n', str(seq))
    #
    # # seq = CCTACGGTGCT C
    # assert (str(trueSeq[0]) == 'CTACGGTGCT')
    #
    # reg = Region('chr1', 0, 10)
    # var = Variant('NA12878', 'chr1', 2, 3, 'indel', 'del', 'TG', 'T')
    #
    # reg.variants.append(var)
    # trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    # assert (str(trueSeq[0]) == 'CCTGTGCTCN')
    #
    # reg = Region('chr1', 0, 10)
    # var = Variant('NA12878', 'chr1', 1, 5, 'sv', 'DEL', '', '')
    #
    # # seq = CCTGGTGCTC, NNCTGCTCNN
    # reg.variants.append(var)
    # trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    # assert (str(trueSeq[0]) == 'NNCTGCTCNN')
    #
    # reg = Region('chr1', 3, 10)
    # var = Variant('NA12878', 'chr1', 0, 6, 'sv', 'INV', '', '')
    #
    # # seq = CCTGGTGCTC, CCT GGTGCTC
    # #      CCTGGT GCTC
    #
    # reg.variants.append(var)
    # trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    # assert (str(trueSeq[0]) == 'NTCCGCTCNN')
    #
    # # seq[0:5].seq.tomutable()[::-1][3:] + seq[3:10][5 - 3:]
    #
    # reg = Region('chr1', 0, 10)
    # var = Variant('NA12878', 'chr1', 0, 3, 'sv', 'DUP', '', '')
    #
    # # seq = CCTGGTGCTC,
    # #      CCTCCTGGTGCTC
    #
    # reg.variants.append(var)
    # trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    # assert (str(trueSeq[0]) == 'CTCCTGGTGC')
    #
    # import imp
    # imp.reload(variant_class)
    #
    # reg = Region('chr1', 1, 10)
    # var = Variant('NA12878', 'chr1', 0, 3, 'sv', 'CNV', '', alt='[3,3]')
    #
    # # seq = CCTGGTGCTC, C CT CCTCCTGGTG CTC
    #
    # # CT CCTCCTGGTG CTC
    # #      CCT CCTCCTGGTG CTC
    #
    # reg.variants.append(var)
    # trueSeq = variant_class.infer_true_seq(reg, seq, segment_size=10)
    # assert (str(trueSeq[0]) == 'CCTCCTGGTG')


def mutate_indel(bou, var, seq, seq_record):
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

    #seq = seq_record[bou.start: bou.end].seq.tomutable()

    if bou.overlap(var) < 0:  # == 0 is acceptable because var.start == var.end in some format
        warnings.warn('Mutation and boundary are not overlapped, boundary: {}, var: {}'.format(str(bou), str(var)))
        return seq

    vt = var.vt

    if vt != 'indel':
        sys.stderr.write('This is not indel\n')

    svtype = var.subtype

    ref = var.ref

    start = var.start
    end = var.end

    alt_list = flatten(var.alt)
    alt = ''
    if len(alt_list) == 1:
        alt = str(alt_list[0])
    elif len(alt_list) > 1:
        alt = str(alt_list[0]) if len(str(alt_list[0])) > 0 else str(alt_list[1])

    #print('alternative:{}'.format(alt))

    # if len(alt) == 0 and svtype != 'del' :
    #     sys.stderr.write('Alternative is blank, check variant: {}'.format(str(var)))

    offset = start - bou.start  # start of the variant relative to the beginning of boundary

    # if offset <= 0:
    #     warnings.warn('-----Ignore this variant:{}, {}'.format(str(var), str(bou)))

    if offset > 0 and seq[offset: min(offset + len(ref), len(seq))] != ref[: min(offset + len(ref), len(seq)) - offset]:

        warnings.warn("offset:{}, len seq: {}, var: {}".format(offset, len(seq), str(var)))
        warnings.warn('reference seq of this boundary: {} is different from ref in variant: {}'.format(
                                                                seq[offset: min(offset + len(ref), len(seq))],
                                                                ref[: min(offset + len(ref), len(seq)) - offset]))

        warnings.warn('reference seq (original): {} should be the same ref in variant: {}'.format(
            str(seq_record[bou.start: bou.end][offset: offset + len(ref)].seq), ref))

        # print('---------------------------\n')
        # print("offset:{}, len seq: {}, var: {}, bou:{}\n".format(offset, len(seq), str(var), str(bou)))
        #
        # print('range1:{} - {}, range2:{} - {}\n'.format(offset, min(offset + len(ref), len(seq)),0,
        #                                                        min(offset + len(ref), len(seq)) - offset))
        # print('seq len:{}, ref len:{}\n'.format(len(seq), len(ref)))
        #
        # print('reference seq of this boundary: {} is different from ref in variant: {}\n'.format(
        #     seq[offset: min(offset + len(ref), len(seq))],
        #     ref[: min(offset + len(ref), len(seq)) - offset]))
        #
        # print('ref:{}, alt:{}\n'.format(ref, alt))
        #
        # print('reference seq (original): {} should be the same ref in variant: {}\n'.format(
        #     str(seq_record[bou.start: bou.end][offset: offset + len(ref)].seq), ref))
        # print('----------------------------\n')


    if vt == 'indel' and offset > 0:

        if svtype == 'ins':
            # first, delete REF, and then insert ALT in place of REF
            if offset + len(ref) >= len(seq):  # if del ends beyond boundary
                seq = seq[0:offset] + alt
            else:
                seq = seq[0:offset] + alt + seq[offset + len(ref): len(seq)]

        elif svtype == 'del' or svtype == 'unknown':

            # delete first
            if len(ref) > 0:
                dlen = len(ref)
            else:
                dlen = end - start

            # if dlen == 0:
            #     dlen = 1

            if offset + dlen >= len(seq):  # if del ends beyond boundary
                seq = seq[0:offset] + alt
            else:
                seq = seq[0:offset] + alt + seq[offset + dlen: len(seq)]

    return seq


def test_mutate_indel():

    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Alphabet import generic_dna
    from common_object import Boundary, Variant

    seq = SeqRecord(Seq('CCTGGTGCTC', generic_dna))

    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 2, 3, 'indel', 'ins', 'T', 'TAC')

    reg.variants.append(var)

    trueSeq = mutate_indel(reg, var, seq)

    assert (str(trueSeq) == 'CCTACGGTGCTC')

    '''-------------'''
    seq = SeqRecord(Seq('CCTGGTGCTC', generic_dna))

    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 12, 13, 'indel', 'ins', 'T', 'TAC')

    reg.variants.append(var)

    trueSeq = mutate_indel(reg, var, seq)

    assert (str(trueSeq) == 'CCTGGTGCTC')

    '''---------------'''
    seq = SeqRecord(Seq('CCTGGTGCTC', generic_dna))
    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 2, 2, 'indel', 'ins', 'T', ['TAC', 'A'])

    reg.variants.append(var)

    trueSeq = mutate_indel(reg, var, seq)

    assert (str(trueSeq) == 'CCTACGGTGCTC')

    '''----------------'''
    seq = SeqRecord(Seq('CCTGGTGCTC', generic_dna))
    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 2, 3, 'indel', 'del', 'TG', 'T')

    reg.variants.append(var)
    trueSeq = mutate_indel(reg, var, seq)

    assert (str(trueSeq) == 'CCTGTGCTC')


def mutate_sv(bou, var, seq, seq_record):
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

    if bou.overlap(var) < 0:  # == 0 is acceptable because var.start == var.end in some format
        warnings.warn('Mutation and boundary are not overlapped, boundary: {}, var: {}'.format(str(bou), str(var)))
        return seq

    vt = var.vt
    if vt != 'sv':
        sys.stderr.write('This is not SV\n')

    svtype = var.subtype

    ref = var.ref

    start = var.start
    end = var.end

    alt_list = flatten(var.alt)

    #print('sv: {}\n'.format(var.alt))

    alt = ''
    if len(alt_list) == 1:
        alt = str(alt_list[0])
    elif len(alt_list) > 1:
        alt = str(alt_list[0]) if len(str(alt_list[0])) > 0 else str(alt_list[1])


    # if len(alt) == 0 and sv != 'DEL':
    #     sys.stderr.write('Alternative is blank, check variant: {}'.format(str(var)))

    offset = start - bou.start  # start of the variant relative to the beginning of boundary

    # if offset <= 0:
    #     warnings.warn('-----Ignore this variant:{}, {}'.format(str(var), str(bou)))
    #
    # if offset > 0 and seq[offset: min(offset + end - start, len(seq))] != ref[: min(offset + end - start, len(seq)) - offset]:
    #     warnings.warn('offset:{}, len seq: {}, var: {}'.format(offset, len(seq), str(var)))
    #     warnings.warn('reference seq of this boundary: {} is different from ref in variant: {}'.format(
    #         seq[offset: min(offset + end - start, len(seq))],
    #         ref[: min(offset + end - start, len(seq)) - offset]))
    #
    #     warnings.warn('reference seq (original): {} should be the same ref in variant: {}'.format(
    #         str(seq_record[bou.start: bou.end][offset: offset + end - start].seq), ref))

    if vt == 'sv':
        if svtype == 'DEL':
            offset = max(0, offset)  # DEL starts before boundary starts
            start = max(start, bou.start)  # if DEL start before boundary starts

            if offset + end - start >= len(seq):  # deletion beyond end of boundary
                seq = seq[0:offset]

            else:  # otherwise, chop off the part from start to end, relatively to offset
                seq = seq[0:offset] + seq[offset + end - start: len(seq)]

        elif svtype == 'INV':

            if offset >= 0:
                seq = seq[0:offset] + seq[offset: offset + end - start][::-1] \
                         + seq[offset + end - start:]

            else:  # if variant starts before boundary starts
                '''retrieve the sequence from the reference seq
                'reverse it and take [: variant.end - bou.start]
                'add it to the rest of seq[j]'''
                seq = seq_record[start:end].seq.tomutable()[::-1][bou.start - start:] + seq[end - bou.start:]


        elif svtype == 'DUP' or svtype == 'DUP:TANDEM':

            if offset >= 0:
                seq = seq[0:offset + end - start] + seq[offset: offset + end - start] + seq[offset + end - start:]

            else:  # if variant starts before boundary starts
                # offset is negative, the portion not in the seq[j]
                seq = seq[0:offset + end - start] + seq_record[start:end].seq.tomutable() + seq[offset + end - start:]

        # ignore for now because CNV often bigger than boundaries
        elif svtype == 'CNV':

            st = alt
            cp = int(re.findall(r'\d+', st)[0])  # extract copy number

            if cp == 0:  # deletion
                offset = max(0, offset)  # DEL starts before boundary starts
                start = max(start, bou.start)  # if DEL start before boundary starts

                if offset + end - start >= len(seq):
                    seq = seq[0:offset]
                else:
                    seq = seq[0:offset] + seq[offset + end - start: len(seq)]

            else:  # insert cp times from the end
                # there is a case when start of variant is before start of the boundary
                # then offset is negative ()

                if end < bou.end:  # otherwise no need to add copy number ? assuming seq[j] is at length of segment_size

                    st = seq_record[start:end].seq.tomutable()

                    offset_end = max(0, offset) + (end - max(bou.start, start))

                    if offset_end < len(seq):
                         seq = seq[:offset_end] + (str(st) * (cp - 1)) + seq[offset_end:]

    return seq


def test_mutate_sv():
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Alphabet import generic_dna
    from common_object import Boundary, Variant

    seq = SeqRecord(Seq('CCTGGTGCTC', generic_dna))

    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 1, 5, 'sv', 'DEL', '', '')

    # seq = CCTGGTGCTC, NNCTGCTCNN
    reg.variants.append(var)
    trueSeq = mutate_sv(reg, var, seq)
    assert (str(trueSeq) == 'CTGCTC')

    '''-----------------------------'''
    seq = SeqRecord(Seq('CCTGGTGCTC', generic_dna))
    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 0, 6, 'sv', 'INV', '', '')

    reg.variants.append(var)
    trueSeq = mutate_sv(reg, var, seq)
    assert (str(trueSeq) == 'TGGTCCGCTC')

    '''-----------------------------'''
    seq = SeqRecord(Seq('CCTGGTGCTC', generic_dna))
    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 0, 3, 'sv', 'DUP', '', '')

    reg.variants.append(var)
    trueSeq = mutate_sv(reg, var, seq)
    assert (str(trueSeq) == 'CCTCCTGGTGCTC')

    '''-----------------------------'''
    seq = SeqRecord(Seq('CCTGGTGCTC', generic_dna))
    reg = Boundary('chr1', 0, 10)
    var = Variant('chr1', 0, 3, 'sv', 'CNV', '', alt='[3,3]')

    reg.variants.append(var)
    trueSeq = mutate_sv(reg, var, seq)
    assert (str(trueSeq) == 'CCTCCTCCTGGTGCTC')




def infer_true_seq(bou, seq_record, segment_size = REGION_SIZE):
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

    # mutate from variants with larger start site so that coordinate of other variants conserved
    # if same start positions, process smaller ones first
    bou.variants = sorted(bou.variants, key=lambda x: (x.chrid, -x.start, (x.end - x.start)))

    # seq_record1 = chrom_list[bou.chrom]
    seq = seq_record[bou.start: bou.end].seq.tomutable()


    for p in range(len(bou.variants)):

        var = bou.variants[p]

        vt = var.vt

        if vt == 'snp':
            seq = mutate_snp(bou, var, seq, seq_record)
        elif vt == 'indel':
            seq = mutate_indel(bou, var, seq, seq_record)
        elif vt == 'sv':
            seq = mutate_sv(bou, var, seq, seq_record)

    seq = padding_truncating(seq, segment_size)

    return seq
