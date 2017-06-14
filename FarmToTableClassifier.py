import re
import csv
import os
import nltk
import string
import pickle
import random

from preprocess import *

import sys

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus.reader.wordnet import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from nltk import FreqDist

c_reader = csv.reader(open('training-data.csv', 'r'),delimiter=',')
examples = list(zip(*c_reader))
documents = examples[1] # sentences or docs are expected in column 1
classes = examples[0] # classes are in column 0

def process_content():
    corpus = []
    vocabularySet = FreqDist()

    for i in range(len(documents)):
        document = documents[i]
        clean_words = preprocess(document)
        for word in clean_words:
            vocabularySet[word] += 1
        corpus.append({'words':clean_words, 'category':classes[i]})

    vocabulary = list(vocabularySet.hapaxes())
    vocabulary.sort()

    return(vocabularySet, vocabulary, corpus)

(vocabSet, vocab, corpus) = process_content()

print("FIRST 10 CORPUS:")
print(corpus[:10])
print("\nFIRST 10 VOCAB :")
print(vocab[0:10])

print("\nfirst category: " + corpus[0]['category'])
print("\nsecond category: " + corpus[1]['category'])
print("first document: " + ', '.join(corpus[0]['words']))
vocab = [item for item in vocabSet if vocabSet[item] >= 5]
vocab.sort()
print(vocab)
print(len(vocab))

random.shuffle(corpus)
featuresets = [(document_features(item['words'], vocab), item['category']) for item in corpus]

print("size of input space " + str(len(featuresets)))
train_set, test_set = featuresets[0:8], featuresets[9:12]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)

save_classifier = open("classifier.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

save_vocab = open("vocab.pickle","wb")
pickle.dump(vocab, save_vocab)
save_vocab.close()