import time, sys
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier

from nltk.tokenize import word_tokenize

def extract_labels(f):
	print('extracting data and labels')
	data = open(f, 'r')
	lines = []
	labels = []
	for line in data:
		line = line.strip('\n')
		line = line.split('\t')
		lines.append(line[0])
		#lines.append(word_tokenize(line[0]))
		labels.append(line[1])
	return lines, labels

def classify_languages(Xtrain, Ytrain, Xtest, X_vk):
	pipeline = Pipeline([
	('features', FeatureUnion([
		('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,5))),
    ])),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam'))
    ])

	print('fitting..')
	pipeline.fit(Xtrain, Ytrain)

	print('predicting..')
	Yguess = pipeline.predict(Xtest)
	print('predicting 2..')
	Yguess_vk = pipeline.predict(X_vk)
	

	return Yguess, Yguess_vk 


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


if __name__ == '__main__':
	start = time.time()
	Xtrain, Ytrain = extract_labels(sys.argv[1])
	Xtest, Ytest = extract_labels(sys.argv[2])
	X_vk, Y_vk = extract_labels(sys.argv[3])

	Xtrain, Ytrain = Xtrain[:100000], Ytrain[:100000]
	
	print('Number of languages:', len(set(Ytest)))
	
	Yguess, Yguess_vk = classify_languages(Xtrain, Ytrain, Xtest, X_vk)
	labels = ["rus", "tat", "che", "bak", "kbd", "bxr", "chv", "mhr", "sah", "inh", "myv", "kpv", "udm", "tyv", "lez", "alt", "krc", "nog", "ava", "ady", "evn", "niv", "lbe", "mdf", "kjh", "xal", "koi", "gld", "ckt", "yrk", "cjs", "tab", "kum", "udi", "abq", "ttt", "dar", "tkr", "yux", "kpy", "kas", "kim", "mns", "aqc", "rut", "kca", "eve", "ykg"]
	#labels = ['tat', 'bak', 'chv', 'sah', 'tyv', 'alt', 'krc', 'nog', 'kjh', 'cjs', 'kum', 'kim', 'che', 'inh', 'lez', 'ava', 'lbe', 'tab', 'udi', 'dar', 'tkr', 'kas', 'aqc', 'rut', 'mhr', 'myv', 'kpv', 'udm', 'mdf', 'koi', 'yrk', 'yux', 'mns', 'kca', 'ykg', 'kbd', 'ady', 'abq', 'evn', 'gld', 'eve', 'bxr', 'xal', 'ckt', 'kpy', 'rus', 'ttt', 'niv']

	print('writing')
	print("\n\nAccuracy: ", accuracy_score(Ytest, Yguess))
	print('\nClassification report:\n', classification_report(Ytest, Yguess, labels=labels))
	cm = confusion_matrix(Ytest, Yguess, labels=labels)

	print_cm(cm, labels)
	print('--------------------VK-----------------------')
	print("\n\nAccuracy: ", accuracy_score(Y_vk, Yguess_vk))
	print('\nClassification report:\n', classification_report(Y_vk, Yguess_vk, labels=labels))
	cm = confusion_matrix(Y_vk, Yguess_vk, labels=labels)

	print_cm(cm, labels)

	end = time.time()
	duration = time.strftime("%H:%M:%S", time.gmtime(end - start))
	print('\nDuration:', duration)