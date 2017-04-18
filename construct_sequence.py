import argparse, model
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('weights', help = 'Location of weights file.')
parser.add_argument('length', help = 'Length of sequence to generate', 
	type = int)
args = parser.parse_args()
mRNN = model.build_model(args.weights)
current_sequence = []
for _ in xrange(args.length):
	tmp = []
	for i in xrange(1, 5):
		tmp.append(current_sequence + [i])
	scores = mRNN.batch_predict(tmp)
	j = np.argmin(scores)
	current_sequence = tmp[j]

lookup = {key : value for key, value in zip(range(1, 5), 'ATCG')}
seq = [lookup[x] for x in current_sequence]
print ''.join(seq)
