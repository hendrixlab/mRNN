# mRNN - mRNA RNN (mRNA recurrent neural network)
mRNN is a package for distinguishing coding transcripts from noncoding using gated recurrent neural networks (GRNNs). 

_How to get mRNN_

You can obtain mRNN through github:

git clone https://github.com/hendrixlab/mRNN.git

_mRNN: Included Files_

1. _Testing and Training_

   train_mRNN.py - training mRNN model based on input training sequences.

   Example: 
   	    $ python mRNN/train_mRNN.py -e 128 -r 32 -d 0.4 -o mRNN.16K.da1.1 -s 3 -E 4 mRNAs.train16K.da1.fa lncRNAs.train16K.da1.fa mRNAs.valid500.fa lncRNAs.valid500.fa > epochs.mRNN.16K.da1.1.txt

   test_mRNN.py

   Example: 
   	    $ python mRNN/test_mRNN.py -w w14u3.pkl -f test mRNAs.test500.fasta lncRNAs.test500.fasta

   test_ensemble_mRNN.py - Ensemble testing with mRNN models. Models are provided as a comma-separated list. 

   Example: 
   	    $ python mRNN/test_ensemble_mRNN.py -w w10u5.pkl,w14u3.pkl,w16u5.pkl,w18u1b.pkl,w23u2.pkl -f test mRNAs.test500.fasta lncRNAs.test500.fasta

   mRNN.py - Basic prediction of coding probability for a set of input sequences. 

   mRNN_ensemble.py - Basic prediction of coding probability for a set of input sequences using a set of models. 

2. _Further Analysis_

   mutation_analysis.py - For the provided input sequence, perform a mutation analysis involving every possible point-mutation on the input sequence, and computing its corresponding score change. 

   pair_mutation_analysis.py - For a provided input sequence, computer all possible pairs of mutation i,a,j,b where position i is changed to a and position j is changed to b. 

   shuffle_analysis.py - For the provided input sequences, shuffle each of them and report the scores of the shuffled sequence to the unaltered sequence. 

   truncation_analysis.py - Coding score trajectory. Compute the mRNN coding probability for all truncations of the input sequence from position 1 to i. 

3. _Core mRNN modules_:

These modules aren't called directly, but involved in basic function of mRNN

      fasta.py - input reading FASTA files

      model.py - building the mRNN model

      evaluate.py - Computing accuracy, batch testing

      preprocessing.py - Utilities for various preprocessing
