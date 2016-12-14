import preprocess
from rnnnumpy import RNNNumpy
import numpy as np
dat = []
data = preprocess.dataset()
rnn = RNNNumpy(data.vocabulary_size)
x = data.X_train[0:100]
o, s = rnn.forward_propagation(data.X_train[1:100])
predict = np.argmax(o ,axis=1)
print (predict)
print(data.index_to_word[2495]);
array_of_words = " ".join([data.index_to_word[x] for x in predict ])

print( array_of_words)
