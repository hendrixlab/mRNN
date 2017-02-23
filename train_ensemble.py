import fasta, preprocessing, model, evaluate
import sys, os, getopt
import inspect

#########
# USAGE #
#########

'''
Prints the usage statement and all options
'''
    
def usage():
    script = os.path.basename(__file__)
    print "\n\nUsage:  " + script + " [options] <positive fasta> <negative fasta>"
    print('''          
          Options:
          
    -h --help\t\tprints this help message.
    -o --output\t\tthe file-base for the output files.
    -w --weights\tpkl file of the model/model weights.
    -E --epochs\tNumber of epochs to train on.(default=50)
    -b --batch_size\tbatch size for testing (default=64)
    -e --embedding_size\tNumber of dimensions in embedding (default=256)
    -r --recurrent_gate_size\tSize of recurrent gate (default=512)
    -d --dropout\tThe dropout probability p_dropout (default=0.4)
    -t --test\tProportion of data to test on. (default=0.1)
    -l --min_length\tminimum length sequence to train on (default=200)
    -L --max_length\tmaximum length sequence to train on (default=1000)
    -s --early_stopping\tNumber of epochs above minimum validation score before stopping
    -m --ensemble\tNumber of models to train for an ensemble (default=1)
''')
    sys.exit()

#########
# MAIN  #
#########

'''
The main loop. Parse input options, run training sequence.
'''
    
def arguments():
    # Options
    opts, files = getopt.getopt(sys.argv[1:], "hvo:w:E:b:e:r:d:t:l:L:s:m:", ["help",
                                                                       "output=",
                                                                       "weights=",
                                                                       "epochs=",
                                                                       "batch_size=",
                                                                       "embedding_size=",
                                                                       "recurrent_gate_size=",
                                                                       "dropout=",
                                                                       "test=",
                                                                       "min_length=",
                                                                       "max_length=",
							               "early_stopping=",
								       "ensemble="
                                                                   ])
    if len(files) != 4:
        usage()
    posFastaFile = files[0]
    negFastaFile = files[1]
    posValFasta = files[2]
    negValFasta = files[3]
    print "using positive file: ", posFastaFile
    print "using negative file: ", negFastaFile
    print "using positive validation file: ", posValFasta
    print "using negative validation file: ", negValFasta
    # Defaults:
    parameters = {}
    parameters['output'] = None
    parameters['verbose'] = False
    parameters['weights'] = None
    parameters['batch_size'] = 16
    parameters['embedding_size'] = 128
    parameters['recurrent_gate_size'] = 256
    parameters['dropout'] = 0.1
    parameters['test'] = 0.1
    parameters['min_length'] = 200
    parameters['max_length'] = 1000
    parameters['num_train'] = 10000
    parameters['epochs'] = 25
    parameters['save_freq'] = 1
    parameters['early_stopping'] = None
    parameters['ensemble'] = 1
    # loop over options:
    for option, argument in opts:
        if option == "-v":
            parameters[verbose] = True
        elif option in ("-h", "--help"):
            usage()
        elif option in ("-o", "--output"):
            parameters['output'] = argument
        elif option in ("-w", "--weights"):
            parameters['weights'] = argument
        elif option in ("-E", "--epochs"):
            parameters['epochs'] = int(argument)
        elif option in ("-b", "--batch_size"):
            parameters['batch_size'] = int(argument)
        elif option in ("-e", "--embedding_size"):
            parameters['embedding_size'] = int(argument)
        elif option in ("-d", "--dropout"):
            parameters['dropout'] = float(argument)
        elif option in ("-t", "--test"):
            parameters['test'] = float(argument)
        elif option in ("-l", "--min_length"):
            parameters['min_length'] = int(argument)
        elif option in ("-L", "--max_length"):
            parameters['max_length'] = int(argument)
        elif option in ("-n", "--num_train"):
            parameters['num_train'] = int(argument)
	elif option in ("-s", "--early_stopping"):
	    if argument is not None:
		argument = int(argument)
	    parameters['early_stopping'] = argument
	elif option in ("-m", "--ensemble"):
	    parameters['ensemble'] = int(argument)
        else:
            assert False, "unhandled option"
    return posFastaFile, negFastaFile, posValFasta, negValFasta, parameters
    ##########
    ## MAIN ##
    ##########

def train(posFastaFile, negFastaFile, posValFasta, negValFasta, parameters):
    print "Reading input files..."
    positives = fasta.load_fasta(posFastaFile,parameters['min_length'])
    negatives = fasta.load_fasta(negFastaFile,parameters['min_length'])
    valpos = fasta.load_fasta(posValFasta,parameters['min_length'])
    valneg = fasta.load_fasta(negValFasta,parameters['min_length'])
    train = positives,negatives
    val = valpos, valneg
    print "Building new model..."
    mRNN = model.build_model(parameters['weights'],parameters['embedding_size'],parameters['recurrent_gate_size'],5,parameters['dropout'])
    print inspect.getmodule(mRNN.__class__)
    print "Training model..."
    mRNN = model.train_model(mRNN, train, val, parameters['epochs'], parameters['output'],parameters['max_length'],parameters['save_freq'],
	parameters['early_stopping'])
    return mRNN

def main():
	args = arguments()
	parameters = args[4]
	files = args[:4]
	output_base = parameters['output']
	best_models = []
	for i in xrange(parameters['ensemble']):
		print output_base
		if output_base:
			new_output = output_base + '.' + str(i)
			parameters['output'] = new_output
		args = files + (parameters,)
		mRNN = train(*args)
		best_epoch = mRNN.valcosts.index(min(mRNN.valcosts))
		print 'Round {0}, best epoch: {1}'.format(i, best_epoch)
		if output_base:
			best_models.append(new_output + '.' + str(best_epoch))
	print best_models
if __name__ == "__main__":
    main()
