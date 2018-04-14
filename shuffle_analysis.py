import fasta, preprocessing, model, evaluate
import sys, os, argparse, re
from scipy.special import logit
from random import shuffle
from numpy import mean, std, arange

'''Usage: truncation_analysis.py <fasta> <model weights file> <output file name>'''
'''Options:
	-o	Overwrite output file (default: error if file already exists)
	-e	Extract Ensembl transcript ID from fasta defline (default: use full defline)
	-n 	Number of shuffles (default 20)
	-p	base of the files where plots will be written. (default: no plot).'''

def main():

	# Options
	parser = argparse.ArgumentParser(description = '''Takes a fasta file and a weights file for an RNN model as input.
		After loading the RNN, the 3' UTR, CDS, and 5' UTR are individually shuffled, and the entire transcript is scored.
		The number of shuffles done is determined with the -n option. The model predicts the probability that the shuffled
		and unshuffled transcripts are protein coding, and Z-scores are computed for the 3' UTR, CDS, and 5' UTR.
		The output is a tab-delimited file with the following fields: transcript name, 3' UTR Z-score, CDS Z-score, 5' UTR Z-score,
		3' UTR length, CDS length, 5' UTR length.''')
	parser.add_argument('fasta', help = '''Fasta file of sequences for shuffling.''')
	parser.add_argument('weights', help = '''File containing model weights. This specifies which model to use.''')
	parser.add_argument('output', help = '''Output name. This is the file where results will be written. 
		By default, this script will not run if the output file already exists. Use the -o option to overwrite an existing file.''')
	parser.add_argument('-o', help = '''Use this option if you want to overwrite an existing output file.''', 
		action = 'store_true')
	parser.add_argument('-e', help = '''Use this option to write just the Ensembl transcript ID instead of 
		the full defline.''', action = 'store_true')
	parser.add_argument('-n', help = '''Number of times to shuffle each segment. Default 20.''', default = 20, type = int)
	parser.add_argument('-p', help = '''Base of filenames to plot to. The plot is a matplotlib plot of z-score 
				vs sequence length. The -o option also applies to this file.''')
	args = parser.parse_args()

	##########
	## MAIN ##
	##########

	if not args.o:
		if os.path.exists(args.output):
			field1 = 'file'
			raise Exception(args.output + ''' already exists! Please choose a different {0} name or use the -o option to overwrite. Use the command python {1} -h for more details.'''.format(field1, sys.argv[0]))
		if args.p and os.path.exists(args.p):
			raise Exception(args.p + ''' already exists! Please choose a different {0} name or use the -o option to overwrite. Use the command python {1} -h for more details.'''.format(field1, sys.argv[0]))
	orig_dir = os.getcwd()
	if args.e: 
		transpat = re.compile('ENST\d*.\d*')
	#never mind, CDS is sufficient
	cds_loc = re.compile('CDS:(\d+)-(\d+)')
	print "Reading input files..."
	full_seqs = fasta.load_fasta(args.fasta, 0)
	seqs = []
	seq_lens = []
	for seq, name in full_seqs:
		coords = cds_loc.search(name)
		#do shuffling here			
		if coords:
			#there is a CDS field in the defline
			coords = map(int, coords.group(1, 2))
		else:
			print name
			continue
		utr5 = seq[:coords[0] - 1]
		cds = seq[coords[0] - 1 : coords[1]]
		utr3 = seq[coords[1]:]
		seqs.append((seq, 'orig', name))
		utr5_shuffle = [shuf_utr5 + cds + utr3 for shuf_utr5 in shuffle_seq(utr5, args.n)]
		cds_shuffle = [utr5 + shuf_cds + utr3 for shuf_cds in shuffle_seq(cds, args.n)]
		utr3_shuffle = [utr5 + cds + shuf_utr3 for shuf_utr3 in shuffle_seq(utr3, args.n)]
		for seq_type, group in zip(('utr5', 'cds', 'utr3'), (utr5_shuffle, cds_shuffle, utr3_shuffle)):
			for s in group:
				seqs.append((s, seq_type, name))
		seq_lens.append(map(len, [utr5, cds, utr3]))
	
	mRNN = model.build_model(args.weights)
	print "Evaluating sequences..."
	seqs, seq_type, name = zip(*seqs)
	cds_coords = []
	if args.e:
		names = [transpat.search(n).group() for n in name]
	else:
		names = [line.strip() for line in name]
	probs = mRNN.batch_predict(seqs)
	
	#delete sequences here, since they are not needed anymore
	del seqs

	logodds = logit(probs)
	Zscores = []
	#calculate z-scores
	i = 0
	while i < len(probs):
		batch = []
		curr_name = names[i]
		while curr_name == names[i]:
			batch.append((probs[i], seq_type[i]))
			i += 1
			if i == len(probs):
				break
		assert batch[0][1] == 'orig'
		orig = batch[0][0]
		tmp = {'name' : curr_name, 'utr5' : None, 'cds' : None, 'utr3' : None}
		for j in xrange(3):
			try:
				sub_batch = batch[1 + j * args.n : 1 + (j + 1) * args.n]
				tmp[sub_batch[0][1]] = z_score(orig, zip(*sub_batch)[0])					
			except IndexError:
				pass
		Zscores.append([tmp[key] for key in ['name', 'utr5', 'cds', 'utr3']])
	comments = '#fasta: {0}, weights: {1}, number of shuffles: {2}'.format(args.fasta, args.weights, args.n)
	lines = [comments, "transcript\t5' UTR Z-score\tCDS Z-score\t3' UTR Z-score\t5' UTR length\tCDS length\t3' UTR length"]
	for z, lens in zip(Zscores, seq_lens):
		line = map(str, z + lens)
		lines.append('\t'.join(line))		
	
	with open(args.output, 'w') as out:
		out.write('\n'.join(lines))

	if args.p:
		plot_zscore_scatter(Zscores, seq_lens, args.p) 
		plot_zscore_histogram(Zscores, seq_lens, args.p)	
		
def shuffle_seq(seq, shuffle_num):
	if len(seq) == 0:
		return []
	seqs = []
        for i in xrange(shuffle_num):
		seq_copy = seq[:]
		shuffle(seq_copy)
		seqs.append(seq_copy)
	return seqs 

def z_score(X, sample):
	return (X - mean(sample)) / std(sample)
	
def plot_zscore_scatter(Zscores, lens, figbase):
	import matplotlib
	matplotlib.use('agg')
	from matplotlib import pyplot as plt
	
	names, utr5, cds, utr3 = zip(*Zscores)
	scores = [utr5, cds, utr3]
	#colors = ['#D7191C', '#FDAE61', '#2C7BB6']
        colors = ['red', 'green', 'blue']
        labels = ["5' UTR", "CDS", "3' UTR"]
	lens = zip(*lens)
	for score, length, color, label in zip(scores, lens, colors, labels):
		plt.scatter(length, score, s = 3, color = color, label = label, alpha = 0.5)
	plt.legend(loc="upper left", scatterpoints=1)
        plt.xscale('log')
        plt.yscale('symlog')
	plt.ylabel('Z-score')
	plt.xlabel('Length of shuffled region (nt)')
	plt.savefig(figbase + "_scatter.svg")	
        plt.clf()

def plot_zscore_histogram(Zscores, lens, figbase):
	import matplotlib
	matplotlib.use('agg')
	from matplotlib import pyplot as plt
	
	names, utr5, cds, utr3 = zip(*Zscores)
	scores = [utr5, cds, utr3]
        allScores = utr5 + cds + utr3
        binWidth = 0.5
        print utr5
        print cds
        print utr3
        print allScores
        print min(allScores)
        print max(allScores)
        bins = arange(min(s for s in allScores if s is not None),max(s for s in allScores if s is not None),binWidth)
	#colors = ['#D7191C', '#FDAE61', '#2C7BB6']
        colors = ['red', 'green', 'blue']
	labels = ["5' UTR", "CDS", "3' UTR"]
	lens = zip(*lens)
	for score, length, color, label in zip(scores, lens, colors, labels):
		plt.hist(score, bins=bins, normed = True, color = color, label = label, linewidth = 0, alpha=0.5)
	plt.legend()
	plt.xlabel('Z-score')
	plt.ylabel('Probability Density')
	plt.savefig(figbase + "_histogram.svg")	
	
	
if __name__ == "__main__":
	main()
