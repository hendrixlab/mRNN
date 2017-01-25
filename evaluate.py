import numpy as np
from passage.utils import save, load
from passage.layers import Embedding, GatedRecurrent, Dense, OneHot, LstmRecurrent, Generic
from passage.models import RNN
import random
import sys

'''
evaluate_model

Inputs:
   model - a pre-trained model RNN built from model.py
   test_data - a tuple of positives,negatives
   max_batch_size - the maximum size of a batch

Returns:
   A confusion matrix of the predictions
'''

def evaluate_model(model, test_data, max_batch_size=16):
    positive, negative = test_data
    print "Preparing batches for prediction..."
    pos_batches = prepare_batches(positive,max_batch_size)
    neg_batches = prepare_batches(negative,max_batch_size)
    p_mat = batch_predict(model, pos_batches)
    n_mat = batch_predict(model, neg_batches)
    return [n_mat, p_mat]

'''
prepare_batches
'''

def prepare_batches(data,batch_size):
    length_dictionary = {}
    data = sorted(data, key=lambda tup: len(tup[0]), reverse=False)
    for d in data:
        length = len(d[0])
        if length not in length_dictionary:
            length_dictionary[length] = []
        length_dictionary[length].append(d)
    batches = []
    for length in length_dictionary:
        datas = length_dictionary[length]
        for b in range(0, len(datas), batch_size):
            batches.append(datas[b:b+batch_size])
    return batches

"""
batch_predict

Inputs:
  model - Model to use for prediction
  data - Data set to generate predictions for
  batch_size - Size of each prediction batch

Note: 
  Each batch of data in batches is a list of tuples of dna, name

Returns:
    An array of the counts of negative, positive predictions
"""

def batch_predict(model, batches):
    result = [0.0,0.0] # negative, positive
    print "Making predictions..."
    for b in batches:
        dna, name = zip(*b)
        r =  model.predict(dna)
        for pred in r:
            round_pred = int(round(pred[0]))
            result[round_pred] += 1
    return result

'''
evaluate_multi_model

Inputs:
   models - a list of pre-trained model RNNs built from model.py
   test_data - a tuple of positives,negatives
   max_batch_size - the maximum size of a batch

Returns:
   A confusion matrix of the predictions
'''

def evaluate_multi_model(models, test_data, max_batch_size=16):
    positive, negative = test_data
    print "Preparing batches for prediction..."
    pos_batches = prepare_batches(positive,max_batch_size)
    neg_batches = prepare_batches(negative,max_batch_size)
    p_mat = batch_multi_predict(models, pos_batches)
    n_mat = batch_multi_predict(models, neg_batches)
    return [n_mat, p_mat]


"""
batch_multi_predict

Inputs:
  models - a list of models to use for prediction
  data - Data set to generate predictions for
  batch_size - Size of each prediction batch

Note: 
  Each batch of data in batches is a list of tuples of dna, name

Returns:
    An array of the counts of negative, positive predictions
"""

def batch_multi_predict(models, batches):
    result = [0.0,0.0] # negative, positive
    print "Making predictions..."
    for b in batches:
        dna, name = zip(*b)
        multi_preds = []
        for model in models:
            preds = []
            r =  model.predict(dna)
            for pred in r:
                #round_pred = int(round(pred[0]))
                #preds.append(round_pred)
                preds.append(pred[0])
            multi_preds.append(preds)
        pred_list = zip(*multi_preds)
        for p in pred_list:
            predictions = list(p)
            thisPred = int(round(sum(predictions)/len(predictions)))
            result[thisPred] += 1
            #pred_sum = sum(predictions)
            #if pred_sum >= len(predictions)/float(2):
            #    result[1] += 1
            #else:
            #    result[0] += 1
    return result
                
'''
get_model_errors

Inputs:
   model - a pre-trained model RNN built from model.py
   test_data - a tuple of positives,negatives
   max_batch_size - the maximum size of a batch

Returns:
   A confusion matrix of the predictions
'''

def get_model_errors(model, test_data, max_batch_size=16):
    positive, negative = test_data
    print "Preparing batches for prediction..."
    pos_batches = prepare_batches(positive,max_batch_size)
    neg_batches = prepare_batches(negative,max_batch_size)
    p_err = batch_get_errors(model, pos_batches, 1)
    n_err = batch_get_errors(model, neg_batches, 0)
    return p_err, n_err


"""
batch_get_errors

Inputs:
  model - Model to use for prediction
  data - Data set to generate predictions for
  batch_size - Size of each prediction batch

Note: 
  Each batch of data in batches is a list of tuples of dna, name

Returns:
    An array of the counts of negative, positive predictions
"""

def batch_get_errors(model, batches, label):
    print "Making predictions..."
    err = []
    for b in batches:
        dna, name = zip(*b)
        r =  model.predict(dna)
        for result in zip(r, dna, name):
            pred,thisDNA,thisName = result
            round_pred = int(round(pred[0]))
            if round_pred != label:
                err.append((thisDNA,thisName))
    return err


def process_results(conf_mat,parameters):
    [[TN, FP], [FN, TP]] = conf_mat
    acc = (TP+TN)/(TP+TN+FP+FN)
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    outFile = parameters['weights'] + ".acc.txt"
    if parameters['file_label']:
        outFile = parameters['weights'] + "." + parameters['file_label'] + ".acc.txt"
        
    F = open(outFile,'w')
    F.write("%s\tACC\t%.4f\n" % (parameters['weights'],acc))
    F.write("%s\tSPEC\t%.4f\n" % (parameters['weights'],spec))
    F.write("%s\tSENS\t%.4f\n" % (parameters['weights'],sens))
    F.close()
    return acc
