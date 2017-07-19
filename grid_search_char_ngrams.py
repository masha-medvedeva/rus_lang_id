from __future__ import print_function
import glob
import os
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from statistics import mean
from datetime import datetime
from nltk.tokenize.casual import TweetTokenizer
from time import time
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
warnings.filterwarnings("ignore", category=UserWarning)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char')),
    ('clf', LinearSVC())
])

parameters = {
    'tfidf__ngram_range': ((1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,2),(2,3),(2,4),(2,5),(2,6),(3,3),(3,4),(3,5),(3,6),(4,4),(4,5),(4,6),(5,5),(5,6),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7)),
    'tfidf__lowercase': (True, False),
    'tfidf__max_df': (0.01, 1.0), # ignore words that occur as more than 1% of corpus
    #'tfidf__min_df': (1, 2, 3), # we need to see a word at least (once, twice, thrice)
    #'tfidf__use_idf': (False, True),
    #'tfidf__sublinear_tf': (False, True),
    'tfidf__binary': (False, True),
    'tfidf__norm': (None, 'l1', 'l2'),
    'clf__C':(1, 5)
}


def extract_labels(f):
    print('extracting data and labels')
    data = open(f, 'r')
    lines = []
    labels = []
    for line in data:
        line = line.split('\t')
        lines.append(line[0])
        labels.append(line[1])
    return lines, labels

if __name__ == "__main__":

    #load datasets
    Xtrain, Ytrain = extract_labels('train.txt')
    Xtrain, Ytrain = Xtrain[:100000], Ytrain[:100000]

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=24, verbose=2) #check n_jobs for available cores
    t0 = time()
    grid_search.fit(Xtrain, Ytrain)


    print("done in %0.3fs" % (time() - t0))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
