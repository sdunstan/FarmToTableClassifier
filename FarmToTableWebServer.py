from flask import Flask
from flask import request
from flask import jsonify
from preprocess import *

import pickle
import nltk

app = Flask(__name__)

classifier_f = open("classifier.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

vocab_f = open("vocab.pickle", "rb")
vocab = pickle.load(vocab_f)
vocab_f.close()

@app.route('/focus-area', )
def classifyFocusArea():
    phrase = request.args.get('PHRASE')

    words = preprocess(phrase)

    features = document_features(words, vocab)
    prob = classifier.prob_classify(features)
    label = classifier.classify(features)
    dist = [{'category':d, 'p': round(prob.prob(d), 7)} for d in prob.samples() if prob.prob(d) >= 0.02]
    sorted(dist, key=lambda item: item['p'])
    result = {'phrase': phrase, 'words': words, 'category': label, 'dist': dist}
    return jsonify(**result)

if __name__ == "__main__":
    app.run(port=9009)
