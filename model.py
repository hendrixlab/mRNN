import numpy as np
from passage.utils import save
from passage.layers import Embedding, GatedRecurrent, Dense, OneHot
from passage.models import RNN as oldRNN
from passage.preprocessing import LenFilter, standardize_targets
from passage.updates import Adadelta
import random
import sys
from theano import function, tensor
#import theano.sandbox.cuda.basic_ops as sbcuda
from time import time
from fasta import tokenize_dna
import cPickle
from os.path import exists

def load(path):
    from passage import layers
    model = cPickle.load(open(path))
    model_class = RNN
    model['config']['layers'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['layers']]
    '''
    results = []
    for layer in model['config']['layers']:
        print(layer)
        result = getattr(layers, layer['layer'])(**layer['config'])
        results.append(result)
        model['config']['layers'] = result
    '''
    model = model_class(**model['config'])
    return model

class RNN(oldRNN):
        def __init__(self, **kwargs):
                self.emb = kwargs['embedding_size']
                del kwargs['embedding_size']
                oldRNN.__init__(self, **kwargs)
		self.settings['embedding_size'] = self.emb
                valY = tensor.dvector('valY')
                Ypredict = tensor.dvector('Ypredict')
                valcost = self.cost(valY, Ypredict)
                self.valcost = function([valY, Ypredict], valcost)
                self.thresholds = np.linspace(.5, .8, 8)
        def val_loss_accuracy(self, preds, labels):
                loss = self.valcost(labels, np.asarray(preds))
                preds = np.asarray(preds)
                labels = np.asarray(labels)
                pos = preds[np.where(labels == 1)]
                neg = preds[np.where(labels == 0)]
                truepos = [pos >= threshold for threshold in self.thresholds]
                trueneg = [neg < threshold for threshold in self.thresholds]
                sens = [sum(x) * 1.0 / len(pos) for x in truepos]
                spec = [sum(x) * 1.0 / len(neg) for x in trueneg]
                return loss, sens, spec
        
        def batch_by_memory(self, seqs):
                #gpu_mem = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
                #max_size = (gpu_mem / self.emb) / 64
                max_size = 256
                lengths = [len(seq) for seq in seqs]
                seqs = zip(lengths, range(len(seqs)), seqs)
                #separate into batches so that memory is not exceeded
                seqs.sort()
                seqs.reverse()
                batches = []
                maxlen = seqs[0][0]
                while maxlen * len(seqs) > max_size:
                        n = max_size / maxlen
                        batches.append(seqs[:n])
                        seqs = seqs[n:]
                        maxlen = seqs[0][0]
                batches.append(seqs)
                return batches
        def batch_predict(self, seqs, verbose = False):
                preds = []
		count = 0
                for batch in self.batch_by_memory(seqs):
                        if verbose:
				count += len(batch)
				print 'Analyzing {0} of {1}'.format(count, len(seqs))
			p = self.predict([x[2] for x in batch])
                        #predictions
                        p = [x[0] for x in p]
                        #indices
                        i = [x[1] for x in batch]
                        preds.extend(zip(i, p))
                preds.sort()
                preds = [x[1] for x in preds]
                return preds

        def fit(self, trX, trY, valX, valY, n_epochs=1, early_stopping = None, len_filter=LenFilter(max_len = 10000, percentile = 100), snapshot_freq=1, path=None):
                """Train model on given training examples and return the list of costs after each minibatch is processed.
                Args:
                  trX (list) -- Inputs
                  trY (list) -- Outputs
                  valX (list) -- Validation Inputs
                  valY (list) -- Validation Outputs
                  n_epochs (int, optional)  -- number of epochs to train for (default 1)
                  early_stopping -- number of consecutive epochs above minimum validation before stopping (default None; no early stopping)
                  len_filter (object, optional) -- object to filter training example by length (default LenFilter())
                  snapshot_freq (int, optional) -- number of epochs between saving model snapshots (default 1)
                  path (str, optional) -- prefix of path where model snapshots are saved.
                        If None, no snapshots are saved (default None)
                Returns:
                  list -- costs of model after processing each minibatch
                """
                if len_filter is not None:
                        trX, trY = len_filter.filter(trX, trY)
                trY = standardize_targets(trY, cost=self.cost)
                n = 0.
                stats = []
                t = time()
                costs = []
                valY = np.asarray(valY)
                self.valcosts = []
                sensitivity = []
                specificity = []
                training_costs = []

                min_val = float('inf')
                min_train = float('inf')
                stopping_count = 0

                for e in range(n_epochs):
			weights = []
                        epoch_costs = []
                        for xmb, ymb in self.iterator.iterXY(trX, trY):
                                c = self._train(xmb, ymb)
                                epoch_costs.append(c)
                                n += len(ymb)
                                if self.verbose >= 2:
                                        n_per_sec = n / (time() - t)
                                        n_left = len(trY) - n % len(trY)
                                        time_left = n_left/n_per_sec
                                        sys.stdout.write("\rEpoch %d Seen %d samples Avg cost %0.4f Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time_left))
                                        sys.stdout.flush()
				weights.append('Epoch: {0} samples: {1} avg cost: {2}'.format(e, n, np.mean(epoch_costs[-250:]))) 
				for layer in self.settings['layers']:
					try:
						w = layer['config']['weights']
					except TypeError:
						w = [p.get_value() for p in layer.params]
					for p in w:			
						weights.append(str(p))
						if np.any(np.isnan(p)):
							err_file = 'error_0.txt'
							i = 1
							while exists(err_file):
								err_file = 'err_{0}.txt'.format(i)
								i += 1
							with open(err_file, 'w') as out:
								out.write('\n'.join(weights))
							raise Exception('NaN weights')
                        costs.extend(epoch_costs)
                        training_costs.append(np.mean(epoch_costs))
			status = "Epoch %d Seen %d samples Avg cost %0.4f Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time() - t)
                        if self.verbose >= 2:
                                sys.stdout.write("\r"+status) 
                                sys.stdout.flush()
                                sys.stdout.write("\n")
                        elif self.verbose == 1:
                                print status
                        if path and e % snapshot_freq == 0:
                                save(self, "{0}.{1}".format(path, e))
                        preds = self.batch_predict(valX)
                        val_loss, sens, spec = self.val_loss_accuracy(preds, valY)

                        print "Validation loss:", val_loss
                        print "Sensitivity:", [round(x, 4) for x in sens]
                        print "Specificity:", [round(x, 4) for x in spec]
                        self.valcosts.append(val_loss)
                        sensitivity.append(sens)
                        specificity.append(spec)
                        #early stopping
                        if early_stopping is not None:
                                if val_loss <= min_val:
                                        min_val = val_loss
                                        #reset count
                                        stopping_count = 0
                                        #keep track of the traning cost at the minimum validation loss
                                        min_train = training_costs[-1]
                                elif training_costs[-1] < min_train:
                                        #only increase counter if the latest training loss is below the training loss at minimum validation loss
                                        stopping_count += 1
                                        if stopping_count >= early_stopping:
                                                break                           
                return training_costs, self.valcosts, sensitivity, specificity


def build_model(weights=None, embedding_size=128, recurrent_gate_size=256, n_features=5, dropout=0.1):
    """
    build_model

    Inputs:
        weights - Path to a weights file to load, or None if the model should be built from scratch
        embedding_size - Size of the embedding layer
        recurrent_gate_size - Size of the gated recurrent layer
        n_features - Number of features for the embedding layer
        dropout - Dropout value

    Returns:
        A model object ready for training (or evaluation if a previous model was loaded via `weights`)
    """
    # vvvvv
    #Modify this if you want to change the structure of the network!
    # ^^^^^
    model_layers = [
        Embedding(size=embedding_size,n_features=n_features),
        GatedRecurrent(size=recurrent_gate_size, p_drop=dropout),
        Dense(size=1, activation='sigmoid', p_drop=dropout)
    ]
    args = {'layers' : model_layers, 'cost' : 'BinaryCrossEntropy', 'verbose' : 2, 'updater': Adadelta(lr=0.5),
            'embedding_size' : embedding_size}
    model = RNN(**args)
    if weights: #Just load the provided model instead, I guess?
        print "Loading previously created weights file: ", weights
        model = load(weights)
    return model

def train_model(model, train_data, val_data, epochs, save_name, max_length, save_freq, early_stopping = None):
    """
    train_model
    Inputs:
        model - Model object to train
        train_data - Dataset to use during training
        epochs - Number of epochs to train for
        save_name - Prefix for output checkpoint models
    """
    #TODO make sure we are still keeping track of transcript names
    positive, negative = train_data
    val_pos, val_neg = val_data
    #Add explicit labels to positive/negative datasets so we
    #can concat them together without losing info
    positive = label_data(positive, 1)
    negative = label_data(negative, 0)
    val_pos = label_data(val_pos, 1)
    val_neg = label_data(val_neg, 0)
    all_data = positive+negative
    all_val = val_pos + val_neg
    tokens, labels = zip(*all_data)
    valX, valY = zip(*all_val)
    # temp
    run_info = model.fit(tokens, labels, valX, valY, n_epochs=epochs, path=save_name, snapshot_freq=save_freq, len_filter=LenFilter(max_len=max_length, percentile=100),
	early_stopping = early_stopping)
    return model

def label_data(data, label):
    """
    label_data

    Inputs:
        data - The data point to convert
        label - The label to pair with the data point

    Returns:
        A tuple with the raw data and its label as separate entries.
    """
    return [(x[0], np.asarray(label)) for x in data]
