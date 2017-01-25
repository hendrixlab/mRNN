import fasta, preprocessing, model, evaluate
import sys, os, getopt

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
    -w --weights\tcomma-separated list of pkl files of the model/model weights.
    -E --epochs\tNumber of epochs to train on.(default=100)
    -b --batch_size\tbatch size for testing (default=64)
    -e --embedding_size\tNumber of dimensions in embedding (default=256)
    -r --recurrent_gate_size\tSize of recurrent gate (default=512)
    -d --dropout\tThe dropout probability p_dropout (default=0.4)
    -t --test\tProportion of data to test on. (default=0.1)
    -l --min_length\tminimum length sequence to train on (default=200)
    -L --max_length\tmaximum length sequence to train on (default=1000)
    -f --file_label\tA text label on the accuracy output files.
    
''')
    sys.exit()

#########
# MAIN  #
#########

'''
The main loop. Parse input options, run training sequence.
'''
    
def main():
    # Options
    opts, files = getopt.getopt(sys.argv[1:], "hvo:w:E:b:e:r:d:t:p:f:", ["help",
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
                                                                         "file_label=",
                                                                     ])
    if len(files) != 2:
        usage()
        
    posFastaFile = files[0]
    negFastaFile = files[1]
    print "using positive file: ", posFastaFile
    print "using negative file: ", negFastaFile
 
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
    parameters['epochs'] = 50
    parameters['save_freq'] = 3
    parameters['file_label'] = ""

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
        elif option in ("-f", "--file_label"):
            parameters['file_label'] = argument
        else:
            assert False, "unhandled option"

    ##########
    ## MAIN ##
    ##########

    print "Reading input files..."
    positives = fasta.load_fasta(posFastaFile,parameters['min_length'])
    negatives = fasta.load_fasta(negFastaFile,parameters['min_length'])
    test = positives,negatives
    if not parameters['weights']:
        print "No weights given with -w parameter.\n"
        sys.exit()
    modelFiles = parameters['weights'].split(',')
    models = []
    for modelFile in modelFiles:
        print "Building model..."
        mRNN = model.build_model(modelFile,parameters['embedding_size'],parameters['recurrent_gate_size'],5,parameters['dropout'])
        models.append(mRNN)    
    print "Evaluating model..."
    conf_mat =  evaluate.evaluate_multi_model(models, test, parameters['batch_size'])
    [[TN, FP], [FN, TP]] = conf_mat
    acc = (TP+TN)/(TP+TN+FP+FN)
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    outFile = "multi_predict" + ".acc.txt"
    if parameters['file_label']:
        outFile = "multi_predict" + "." + parameters['file_label'] + ".acc.txt"
        
    F = open(outFile,'w')
    F.write("%s\tACC\t%.4f\n" % (parameters['weights'],acc))
    F.write("%s\tSPEC\t%.4f\n" % (parameters['weights'],spec))
    F.write("%s\tSENS\t%.4f\n" % (parameters['weights'],sens))
    F.close()

if __name__ == "__main__":
    main()
