from selfpeptide.utils.constants import *


def extract_peptides_from_protein_sequence(protein_seq, l=9):
    if len(protein_seq)<l:
        return []
    peptides = []
    for i in range(len(protein_seq)-l+1):
        peptides.append(protein_seq[i:i+l])
    return peptides


def get_vocabulary_tokens():
    return amino_acids + [ALIGNMENT_TOKEN, PADDING_TOKEN]