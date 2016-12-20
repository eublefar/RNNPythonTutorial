import preprocess
from rnnnumpy import RNNNumpy
import numpy as np
dat = []
data = preprocess.dataset()
rnn = RNNNumpy(data.vocabulary_size)
x = data.X_train[10]
predict = rnn.predict(x)

print("predict shape = " + str(predict.shape))
print (unicode(predict))
array_of_words = " ".join([data.index_to_word[x] for x in predict ])

print( array_of_words)
print "Expected Loss for random predictions: %f" % np.log(data.vocabulary_size)
print "Actual loss: %f" % rnn.calculate_loss(data.X_train[:1000], data.Y_train[:1000])
