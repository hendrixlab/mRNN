import fasta, preprocessing, model, evaluate
import sys, os, argparse, re

'''Usage: mutation_analysis.py <fasta> <model weights file> <output file name>'''
'''Options:
	-o	Overwrite output file (default: error if file already exists)
	-e	Extract Ensembl transcript ID from fasta header (default: use full header)'''

def main():

	# Options
	parser = argparse.ArgumentParser()
	parser.add_argument('fasta', help = '''Fasta file of sequences for truncation.''')
	parser.add_argument('weights', help = '''File containing model weights. This specifies which model to use.''')
	parser.add_argument('output', help = '''File where results will be written. By default, this script will
		not run if the output file already exists. Use the -o option to overwrite an existing file.''')
	parser.add_argument('-o', help = '''Use this option if you want to overwrite an existing output file.''', 
		action = 'store_true')
	parser.add_argument('-e', help = '''Use this option to write just the Ensembl transcript ID instead of 
		the full header.''', action = 'store_true')
	args = parser.parse_args()

	##########
	## MAIN ##
	##########
	lookup = dict(zip(range(5), 'NATCG'))
	if not args.o and os.path.exists(args.output):
		raise Exception(args.output + ' already exists!')
	if args.e: 
		transpat = re.compile('ENST\d*.\d*')
	print "Reading input files..."
	full_seqs = fasta.load_fasta(args.fasta, 0)
	mut = []
	for seq, name in full_seqs:
		for i in xrange(len(seq)):
			for b in xrange(1, 5):
				if b != seq[i]:
					mut_seq = seq[:i] + [b] + seq[i + 1:]
					mut.append((mut_seq, str(i), lookup[b], name))
	mRNN = model.build_model(args.weights)
	lines = []
	print "Evaluating sequences..."
	seqs, pos, base, name = zip(*mut)
	if args.e:
		name = [transpat.search(n).group() for n in name]
	scores = mRNN.batch_predict(seqs, True)
	scores = map(str, scores)
	lines = zip(name, pos, base, scores)
	lines = ['\t'.join(line) for line in lines]
	with open(args.output, 'w') as out:
		out.write('\n'.join(lines))
		
        

if __name__ == "__main__":
	main()
