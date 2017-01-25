import random

"""
preprocessing -- 
    This module contains functions vital for the generation of a good 
    training and testing dataset.
"""

def gen_test_train(positive, negative, pct):
    """
    gen_test_train

    Inputs:
        one - 'Positive' dataset
        two - 'Negative' dataset
        amt - Portion of the dataset to pull as the test set (0-1)

    Returns:
        ((positive_train, negative_train), (positive_test, negative_test))
        A tuple containing the training and testing datasets, each a 
        tuple containing ther respective positive and negative datasets.
    """

    def single_test_train(data, pct):
        total = len(data)
        test_amt = int(pct*total)
        idx = range(total)
        random.shuffle(idx)
        test_idx = idx[:test_amt]
        train_idx = idx[test_amt:]
        return (_index_by(data, train_idx), _index_by(data, test_idx))

    p_train, p_test = single_test_train(positive, pct)
    n_train, n_test = single_test_train(negative, pct)
    return ((p_train, n_train), (p_test, n_test))

def sample_even_length(data, num):
    """
    sample_even_length -- Samples from two distributions such that the resulting distributions 
        share a similar lengthdistribution

    Inputs:
        one - 'Positive' dataset
        two - 'Negative' dataset
        num - Total number of samples to include in the result.

    Note:  
        both positive and negative is a list of tuples: dna, name

    Returns:
        (positive, negative)
        A tuple containing the re-sampled positive and negative
        datasets. Each of the two resultant datasets will have num/2 
        elements.  The contents of the two resultant datasets will 
        have similar length distributions.
    """
    positive, negative = data
    if len(positive) < len(negative):
        swapped = True
        shorter = positive
        longer = negative
    else:
        swapped = False
        shorter = negative
        longer = positive
    #Sort the samples (in the longer set) by their length
    #Record the lengths of the shorter ones
    # since x[0] is the dna, in the tuple, we add len(x[0])
    short_lens = zip([len(x[0]) for x in shorter], shorter)
    long_lens = zip([len(x[0]) for x in longer], longer)
    long_lens.sort()
    
    #Get num/2 random samples from the shorter dataset
    idxs = range(len(short_lens))
    random.shuffle(idxs)
    idxs = idxs[:num/2]
    if num/2 > len(short_lens):
        print "ERROR" #TODO
        return None
    filtered_short = _index_by(short_lens, idxs)

    #Get an equal number of samples from the longer dataset
    #that are similar in length to those in the shorter.
    filtered_long = []
    for d in filtered_short:
        length, _ = d
        for e in long_lens:
            long_len, _ = e
            if long_len >= length:
                to_add = e
                break
            else:
                to_add = long_lens[-1]
            filtered_long.append(to_add)
            long_lens.remove(to_add)

    #Remove the length info now that we've evened the length
    #distributions. 
    #x[1] should be the original name, dna tuple
    final_short = [x[1] for x in filtered_short]
    final_long = [x[1] for x in filtered_long]

    if swapped:
        return final_short, final_long
    else:
        return final_long, final_short

def filter_length(data, max_len=1000, min_len=200):
    """
    filter_length 
    
    Inputs:
        data - list of sequence/name tuples. The tuples are arbitrary,
            but the first element must be sequences..
        max_len - maximum length to include in the dataset [1000]
        min_len - minimum length to include in the dataset [200]

    Note:  
       input data is a list of tuples: dna, name

    Returns:
        tuple of data that is filtered by length.
    """
    return filter(lambda x: len(x[0]) >= min_len and len(x[0]) < max_len, seqs)


def _index_by(data, index):
    """
    __index_by

    Inputs:
        data - whatever data
        index - list of indices to pull

    Returns:
        A list of each element in data at the corresponding 
        indices in index.
    """
    return [data[i] for i in index]

def sampleData(data,frac):
    length = len(data)
    samples = random.sample(data, int(frac*length))
    return samples

def generateMutatedData(data):
    newData = []
    for record in data:
        dna,name = record
        newName = name + "|mut"
        newDna = dna
        r = random.randrange(0,len(dna))
        b = dna[r]
        rB = b
        while(rB == b):
            rB = random.randrange(1,5)
        newDna[r] = rB
        newData.append((newDna,newName))
    return newData
