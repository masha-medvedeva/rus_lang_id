import os
import sys
import time
import itertools
import prep
import time
import re

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

from sklearn.base import TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def identity(x):
	return list(itertools.chain.from_iterable(x))


def classify_groups(Xtrain, Ytrain, Xtest):
	pipeline = Pipeline([
	('features', FeatureUnion([
		('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,4))),
		#('wordvec', TfidfVectorizer(analyzer = 'word', binary=True)),
    ])),
    ('classifier', LinearSVC())
    ])
	print('fitting groups..')
	pipeline.fit(Xtrain, Ytrain)
	Yguess = pipeline.predict(Xtest)
	return Yguess

def languages_in_groups_train(group, Xtrain, Ytrain, Ztrain):
	#print(len(Ytrain), len(Ztrain))
	group_lines = []
	group_labels = []
	for i in range(len(Ytrain)):
		if Ytrain[i] == group:
			group_lines.append(Xtrain[i])
			group_labels.append(Ztrain[i])
	return group_lines, group_labels

def languages_in_groups_test(group, Xtrain, Ytrain):
	group_lines = []
	group_labels = []
	for i in range(len(Ytrain)):
		if Ytrain[i] == group:
			group_lines.append(Xtrain[i])
			#group_labels.append(Ztrain[i])
	return group_lines, group_labels

def classify_within_groups(group, Xtrain, Ytrain, Ztrain, Xtest, Ytest_guess):
	print("GROUP:", group)
	Xtrain_group, Ztrain_group = languages_in_groups_train(group, Xtrain, Ytrain, Ztrain)
	Xtest_group, Ztest_group= languages_in_groups_train(group, Xtest, Ytest_guess, Ztest)

	if group == 'TRK':
		pipeline = Pipeline([
		('features', FeatureUnion([
			('charvec', TfidfVectorizer(analyzer = 'char', binary=True, lowercase=False, ngram_range = (1,6))),
		])),
		('classifier', LinearSVC(C=5))
		])
		print('fitting..')
		pipeline.fit(Xtrain_group, Ztrain_group)
		Zguess_group = pipeline.predict(Xtest_group)

	elif group == 'UR':
		pipeline = Pipeline([
		('features', FeatureUnion([
			('charvec', TfidfVectorizer(analyzer = 'char', binary=True, lowercase=False, ngram_range = (1,6), max_df=0.01)),
		])),
		('classifier', LinearSVC(C=5))
		])
		print('fitting..')
		pipeline.fit(Xtrain_group, Ztrain_group)
		Zguess_group = pipeline.predict(Xtest_group)

	elif group == 'ND':
		pipeline = Pipeline([
		('features', FeatureUnion([
			('charvec', TfidfVectorizer(analyzer = 'char', binary=True, lowercase=True, ngram_range = (1,6))),
		])),
		('classifier', LinearSVC(C=5))
		])
		print('fitting..')
		pipeline.fit(Xtrain_group, Ztrain_group)
		Zguess_group = pipeline.predict(Xtest_group)

	elif group == 'AA':
		pipeline = Pipeline([
		('features', FeatureUnion([
			('charvec', TfidfVectorizer(analyzer = 'char', binary=True, lowercase=False, ngram_range = (1,7))),
		])),
		('classifier', LinearSVC(C=5))
		])
		print('fitting..')
		pipeline.fit(Xtrain_group, Ztrain_group)
		Zguess_group = pipeline.predict(Xtest_group)

	elif group == 'TNG':
		pipeline = Pipeline([
		('features', FeatureUnion([
			('charvec', TfidfVectorizer(analyzer = 'char', binary=True, ngram_range = (1,4))),
		])),
		('classifier', LinearSVC(C=5))
		])
		print('fitting..')
		pipeline.fit(Xtrain_group, Ztrain_group)
		Zguess_group = pipeline.predict(Xtest_group)

	elif group == 'CK':
		pipeline = Pipeline([
		('features', FeatureUnion([
			('charvec', TfidfVectorizer(analyzer = 'char', lowercase=False, ngram_range = (1,3))),
		])),
		('classifier', LinearSVC(C=5))
		])
		print('fitting..')
		pipeline.fit(Xtrain_group, Ztrain_group)
		Zguess_group = pipeline.predict(Xtest_group)

	elif group == 'MNG':
		pipeline = Pipeline([
		('features', FeatureUnion([
			('charvec', TfidfVectorizer(analyzer = 'char', binary=True, lowercase=False, ngram_range = (1,5))),
		])),
		('classifier', LinearSVC(C=5))
		])
		print('fitting..')
		pipeline.fit(Xtrain_group, Ztrain_group)
		Zguess_group = pipeline.predict(Xtest_group)

	elif group == 'IE':
		pipeline = Pipeline([
		('features', FeatureUnion([
			#('wordvec', TfidfVectorizer(analyzer = 'word', binary=True)),
			('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,2))),
		])),
		('classifier', LinearSVC(C=5))
		])
		print('fitting..')
		pipeline.fit(Xtrain_group, Ztrain_group)
		Zguess_group = pipeline.predict(Xtest_group)

	return Zguess_group, Ztest_group

if __name__ == '__main__':
	start2 = time.time()
	sys.stdout=open("FAMILIES"+sys.argv[2],"w")
	Xtrain, Ytrain, Ztrain = prep.extract_words_and_labels(sys.argv[1])
	Xtest, Ytest, Ztest = prep.extract_words_and_labels(sys.argv[2])

	print('Training on', len(Xtrain), 'lines')

	#predicted values of the groups
	groups = ["TRK", "ND", "UR", "AA", "TNG", "MNG", "CK", "IE", "ISO"]

	Ytest_guess = classify_groups(Xtrain, Ytrain, Xtest)
	print("\n\nAccuracy groups: ", accuracy_score(Ytest, Ytest_guess))
	print('\nClassification report:\n', classification_report(Ytest, Ytest_guess, labels=groups))
	print('\n', confusion_matrix(Ytest, Ytest_guess, labels=groups))

	overall_true = []
	overall_pred = []
	sentences_pred = []

	for i in range(len(Xtest)):
		if Ytest_guess[i] == 'ISO':
			overall_true.append(Ytest[i])
			overall_pred.append(Ytest_guess[i])
	print(len(overall_true))
	for group in groups[:-1]:
		Zguess_g, Ztest_g = classify_within_groups(group, Xtrain, Ytrain, Ztrain, Xtest, Ytest_guess)
		overall_true.extend(Ztest_g)
		overall_pred.extend(Zguess_g)
		print("\n\nAccuracy: ", accuracy_score(Ztest_g, Zguess_g))
		print('\nClassification report:\n', classification_report(Ztest_g, Zguess_g))
	print(len(overall_true))
	labels = ['tat', 'bak', 'chv', 'sah', 'tyv', 'alt', 'krc', 'nog', 'kjh', 'cjs', 'kum', 'kim', 'che', 'inh', 'lez', 'ava', 'lbe', 'tab', 'udi', 'dar', 'tkr', 'kas', 'aqc', 'rut', 'mhr', 'myv', 'kpv', 'udm', 'mdf', 'koi', 'yrk', 'yux', 'mns', 'kca', 'ykg', 'kbd', 'ady', 'abq', 'evn', 'gld', 'eve', 'bxr', 'xal', 'ckt', 'kpy', 'rus', 'ttt', 'niv']
	
	print("\n\nOverall accuracy: ", accuracy_score(overall_true, overall_pred))
	cm = confusion_matrix(overall_true, overall_pred, labels=labels)
	print('\nClassification report:\n', classification_report(overall_true, overall_pred, labels=labels))
	print_cm(cm, labels)


	end2 = time.time()
	duration2 = time.strftime("%H:%M:%S", time.gmtime(end2 - start2))
	print('Overall time:', duration2)
	sys.stdout.close()
