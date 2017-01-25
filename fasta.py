
"""
fasta 
"""

def load_fasta(filename, min_size):
    """
    load_fasta

    Inputs:
        filename - name of FASTA file to load 
        min_size - minimum sequence length to accept

    Returns:
        Iterator of tuples containing the name and sequence data
    """
    seqs = list()    
    seq = ""
    with open(filename, 'r') as fd:
        for line in fd:
            if ">" in line: #Any line with a > is a comment/separator
                if seq != "":
                    dna = tokenize_dna(seq)
                    seqs.append((dna, name))
                seq = ""
                name = line
            else:
                seq += line.strip()
    if seq != "":
        dna = tokenize_dna(seq)
        seqs.append((dna, name))
    return seqs

def tokenize_dna(seq):
    """
    tokenize_dna

    Inputs:
        seqs - list of sequences

    Returns:
        List of tokenized sequneces
    """
    lookup = dict(zip('NATCG', range(5)))
    token = [lookup[c] for c in seq.upper()]
    return token
