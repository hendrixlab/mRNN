import fasta, preprocessing, model, evaluate
import sys, os, argparse, re
from scipy.special import logit
from random import shuffle
from numpy import mean, std

'''Usage: truncation_analysis.py <fasta> <model weights file> <output file name>'''
'''Options:
	-o	Overwrite output file (default: error if file already exists)
	-e	Extract Ensembl transcript ID from fasta defline (default: use full defline)
	-n 	Number of shuffles (default 10).'''

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
	parser.add_argument('-n', help = '''Number of times to shuffle each segment. Default 10.''', default = 10, type = int)
	args = parser.parse_args()

	##########
	## MAIN ##
	##########

	if not args.o and os.path.exists(args.output):
		if args.s:
			field1 = 'directory'
		else: 
			field1 = 'file'
		raise Exception(args.output + ''' already exists! Please choose a different {0} name or use the -o option to overwrite. Use the command python {1} -h for more details.'''.format(field1, sys.argv[0]))
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
	lines = ["transcript\t5' UTR Z-score\tCDS Z-score\t3' UTR Z-score\t5' UTR length\tCDS length\t3' UTR length"]
	for z, lens in zip(Zscores, seq_lens):
		line = map(str, z + lens)
		lines.append('\t'.join(line))		
	
	with open(args.output, 'w') as out:
		out.write('\n'.join(lines))
		
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
	
if __name__ == "__main__":
	main()
