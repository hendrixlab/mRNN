import fasta, preprocessing, model, evaluate
from scipy.special import logit
import numpy as np
from os.path import exists

def pairMutate(seq):
	'''Mutates every possible pair of bases. Returns a list of tuples. Each tuple contains the mutated sequence, the positions mutated, and which nucleotide
	each base was changed to.'''
	lookup = dict(zip(range(5), 'NATCG'))

	seqs = []
	for i in xrange(len(seq) - 1):
		for basei in xrange(1, 5):
			if basei == seq[i]:
				continue
			for j in xrange(i + 1, len(seq)):
				for basej in xrange(1, 5):
					if basej == seq[j]:
						continue
					newseq = seq[:]
					newseq[i] = basei
					newseq[j] = basej
					seqs.append((newseq, i, j, lookup[basei], lookup[basej]))
	return seqs

def mutantsFromFasta(inputFile):
	'Loads the first sequence from the given file, mutates it.'
	seq = fasta.load_fasta(inputFile, 0)[0][0]
	seqs = zip(*pairMutate(seq))
	return seq, seqs

if __name__ == '__main__':
	from sys import argv, exit
        usage = "Usage: " + argv[0] + " <fasta> <weights> <output>"
        if len(argv) != 4:
                print usage
                exit(0)
	inputFile, weights, output = argv[1:]
	if exists(output):
		raise Exception('Error: ' + output + ' already exists!')
	print 'Building model...'
	mRNN = model.build_model(weights)
	print 'Mutating...'
	seq, mutseqs = mutantsFromFasta(inputFile)

	#make list of sequences: all of the mutated sequences, with original sequence at the end
	seqs = list(mutseqs[0]) + [seq]	
	print 'Predicting...'
	probs = mRNN.batch_predict(seqs)
	scores = logit(probs)

	#make array of the probabilities of mutated sequences
	mut_probs = probs[:-1]
	#make array consisting only of the probability of the original sequence
	orig_prob = np.asarray([probs[-1] for _ in mut_probs])
	#do the same for scores
	orig_score = np.asarray([scores[-1] for _ in mut_probs])
	mut_scores = scores[:-1]
	dS = mut_scores - scores[-1]
	seq_info = mutseqs[1:] + [orig_prob, mut_probs, orig_score, mut_scores, dS]
	#stringify
	seq_info = [map(str, info) for info in seq_info]
	lines = ['\t'.join(line) for line in zip(*seq_info)]
	#repeat check
	if exists(output):
		raise Exception('Error: ' + output + ' already exists!')
	with open(output, 'w') as out:
		out.write('\n'.join(lines))




