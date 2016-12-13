import preprocess
from rnnnumpy import RNNNumpy
dat = []
data = preprocess.dataset()
rnn = RNNNumpy(data.vocabulary_size)
array_of_indices =[ np.nonzero(x)[0][0]  for x,s in rnn.forward_propagation(data.X_train[0:99])]
array_of_words = [dat.append(data.index_to_word[x]) for x in array_of_indices ]
print( "".join(array_of_words))
