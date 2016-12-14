import csv
import numpy as np
import itertools
import nltk
import pickle
import os

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

class dataset:


    def file_is_empty(self, path):
        with open(path,'rb') as file:
            file.seek(0,2)
            print(file.tell())
            return file.tell() == 0


    def preprocess_data(self):
        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        print "Reading CSV file..."
        with open('data/reddit-comments-2015-08.csv', 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            # Split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            # Append SENTENCE_START and SENTENCE_END
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
        print "Parsed %d sentences." % (len(sentences))

        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print "Found %d unique words tokens." % len(word_freq.items())

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(self.vocabulary_size-1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(unknown_token)
        self.word_to_index = dict([(w,i) for i,w in enumerate(self.index_to_word)])

        print "Using vocabulary size %d." % self.vocabulary_size
        print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in self.word_to_index else unknown_token for w in sent]

        print "\nExample sentence: '%s'" % sentences[0]
        print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

        # Create the training data
        tokenized_words = [item for sublist in tokenized_sentences for item in sublist]
        self.X_train = np.asarray([self.word_to_index[w] for w in tokenized_words[:-1]])
        self.Y_train = np.asarray([self.word_to_index[w] for w in tokenized_words[1:]])

    def __init__(self):
        self.vocabulary_size = 8000


        with open('train.pkl', 'ab') as out_data, open('train.pkl','rb') as in_data:
            if not self.file_is_empty('train.pkl'):
                print("previous records :" + str(not self.file_is_empty('train.pkl')) + "\n Loading data...")
                self.X_train = pickle.load(in_data)
                self.Y_train = pickle.load(in_data)
                self.vocabulary_size = pickle.load(in_data)
                self.index_to_word = pickle.load(in_data)
                self.word_to_index = pickle.load(in_data)
            elif self.file_is_empty('train.pkl'):
                print(" no previous records :" + str(self.file_is_empty('train.pkl')) + "\n Generating Data")
                self.preprocess_data()
                print("Saving data")
                pickle.dump(self.X_train, out_data, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.Y_train, out_data, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.vocabulary_size, out_data, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.index_to_word, out_data, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.word_to_index, out_data, pickle.HIGHEST_PROTOCOL)
                in_data.close()
                out_data.flush()
