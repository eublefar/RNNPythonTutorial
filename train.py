import subprocess
import preprocess
from rnnnumpy import RNNNumpy
import numpy as np
import pickle

dat = []
data = preprocess.dataset()
rnn = RNNNumpy(data.vocabulary_size)
x = data.X_train[20006]

np.random.seed(10)
# Train on a small subset of the data to see what happens

losses = RNNNumpy.train_with_sgd(rnn, data.X_train[:20000], data.Y_train[:20000], nepoch=10, evaluate_loss_after=1)

predict = rnn.predict(x)

print("predict shape = " + str(predict.shape))
print (unicode(predict))
array_of_words = " ".join([data.index_to_word[x] for x in predict ])

print( array_of_words)


with open('model.pkl','wb') as out:
    pickle.dump(rnn, out, pickle.HIGHEST_PROTOCOL)

subprocess.Popen(["git", "add", "model.pkl"], stdout=subprocess.PIPE)
subprocess.Popen(["git", "commit", "-m\"auto commit model.pkl\""], stdout=subprocess.PIPE)
process = subprocess.Popen(["git", "push"], stdout=subprocess.PIPE)
