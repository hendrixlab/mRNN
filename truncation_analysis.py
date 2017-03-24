import fasta, preprocessing, model, evaluate
import sys, os, argparse, re
from scipy.special import logit

'''Usage: truncation_analysis.py <fasta> <model weights file> <output file name>'''
'''Options:
	-o	Overwrite output file (default: error if file already exists)
	-e	Extract Ensembl transcript ID from fasta defline (default: use full defline)
	-s	Split output into individual files, one per transcript (default: one file)'''


def main():

	# Options
	parser = argparse.ArgumentParser(description = '''Takes a fasta file and a weights file for an RNN model as input.
		After loading the RNN, each transcript in the fasta file is truncated at every possible position and the model
		predicts the score. The output is a tab-delimited file with the following fields: transcript name, truncation position,
		the model's prediction that the truncated sequence is coding, the log odds of that probability, and information about
		where the position is in the transcript (5' UTR, CDS, 3' UTR, or none). If the -s
		option is used, the transcipt name is in the filename, so the field is eliminated.''')
	parser.add_argument('fasta', help = '''Fasta file of sequences for truncation.''')
	parser.add_argument('weights', help = '''File containing model weights. This specifies which model to use.''')
	parser.add_argument('output', help = '''Output name. By default, this is the file where results will be written. 
		If using the -s option, it is the directory where results will be written. By default, this script will
		not run if the output file or directory already exists. Use the -o option to overwrite an existing file,
		or to potentially overwrite files in the output directory if using the -s option. Note that with the -s option,
		file names are chosen based on the defline of the transcript. If the -s and -o options are used together,
		files in the output directory will not be deleted, but may (or may not) be overwritten.''')
	parser.add_argument('-o', help = '''Use this option if you want to overwrite an existing output file, or use an 
		existing output directory, potentially (but not certainly) overwriting files in it.''', 
		action = 'store_true')
	parser.add_argument('-e', help = '''Use this option to write just the Ensembl transcript ID instead of 
		the full defline.''', action = 'store_true')
	parser.add_argument('-s', help = '''Use this option to split the output into individual files named based on the 
		defline of the transcript. If using this option, the output argument should be the name of a directory,
		not a file.''', action = 'store_true')
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
	cds_loc = re.compile('CDS:(\d+)-(\d+)')
	print "Reading input files..."
	full_seqs = fasta.load_fasta(args.fasta, 0)
        if args.s:
                try:
                        os.mkdir(args.output)
                except OSError:
                        #os.mkdir raises this error if the path already exists. We should not see this error unless using the args.o option.
                        #if the assertion is true, then everything's ok, and we can just ignore the error and move to the directory since it already exists.
                        assert args.o == True
	trunc = []
	for seq, name in full_seqs:
		coords = cds_loc.search(name)
		if coords:
			#there is a CDS field in the defline
			coords = map(int, coords.group(1, 2))
		for i in xrange(len(seq)):
			if not coords:
				pos_class = 'NA'
			elif i < coords[0] - 1: #subtract one because info in defline is 1-based
				pos_class = 'UTR5'
			elif i < coords[1]: 
				pos_class = 'CDS'
			else:
				pos_class = 'UTR3'
			trunc.append((seq[: i + 1], str(i), name, pos_class))	
	mRNN = model.build_model(args.weights)
	print "Evaluating sequences..."
	seqs, pos, name, pos_class = zip(*trunc)
	cds_coords = []
	if args.e:
		names = [transpat.search(n).group() for n in name]
	else:
		names = [line.strip() for line in name]
	probs = mRNN.batch_predict(seqs)
	logodds = logit(probs)
	#stringify numbers	
	probs = map(str, probs)
	logodds = map(str, logodds)

        if args.s:
                os.chdir(args.output)
		lines = zip(pos, probs, logodds, pos_class)
		lines = ['\t'.join(line) for line in lines]
		#put lines in arrays keyed by transcript name
		linedict = {name : [] for name in set(names)}
		for name, line in zip(names, lines):
			linedict[name].append(line)
		for name in linedict:
			with open(name + '.trunc.txt', 'w') as out:
				out.write('\n'.join(linedict[name]))
		#go back to original directory
		os.chdir(orig_dir)
	else:
	        lines = zip(names, pos, probs, logodds, pos_class)
	        lines = ['\t'.join(line) for line in lines]	
		with open(args.output, 'w') as out:
			out.write('\n'.join(lines))
		
        

if __name__ == "__main__":
	main()
