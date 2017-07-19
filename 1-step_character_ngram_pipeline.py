import time, sys
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_graphviz

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
		labels.append(line[1])
	return lines, labels



def classify_languages(Xtrain, Ytrain, Xtest):
	pipeline = Pipeline([
	('features', FeatureUnion([
		('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,5), norm='l2')),
    ])),
    ('classifier', LinearSVC(C=5))

    ])

	print('fitting..')
	pipeline.fit(Xtrain, Ytrain)
	
	print('predicting..')
	Yguess = pipeline.predict(Xtest)	

	return Yguess


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
	
	print('Number of languages:', len(set(Ytest)))
	
	Yguess = classify_languages(Xtrain, Ytrain, Xtest)
	#labels = ["rus", "tat", "che", "bak", "kbd", "bxr", "chv", "mhr", "sah", "inh", "myv", "kpv", "udm", "tyv", "lez", "alt", "krc", "nog", "ava", "ady", "evn", "niv", "lbe", "mdf", "kjh", "xal", "koi", "gld", "ckt", "yrk", "cjs", "tab", "kum", "udi", "abq", "ttt", "dar", "tkr", "yux", "kpy", "kas", "kim", "mns", "aqc", "rut", "kca", "eve", "ykg"]
	labels = ['tat', 'bak', 'chv', 'sah', 'tyv', 'alt', 'krc', 'nog', 'kjh', 'cjs', 'kum', 'kim', 'che', 'inh', 'lez', 'ava', 'lbe', 'tab', 'udi', 'dar', 'tkr', 'kas', 'aqc', 'rut', 'mhr', 'myv', 'kpv', 'udm', 'mdf', 'koi', 'yrk', 'yux', 'mns', 'kca', 'ykg', 'kbd', 'ady', 'abq', 'evn', 'gld', 'eve', 'bxr', 'xal', 'ckt', 'kpy', 'rus', 'ttt', 'niv']
	print('writing')

	sys.stdout=open("final_model_results_char_"+sys.argv[2]+'.txt',"w")
	print("\n\nAccuracy: ", accuracy_score(Ytest, Yguess))
	print('\nClassification report:\n', classification_report(Ytest, Yguess, labels=labels))
	print('\nClassification report:\n', classification_report(Ytest, Yguess))

	cm = confusion_matrix(Ytest, Yguess, labels=labels)

	print_cm(cm, labels)

	end = time.time()
	duration = time.strftime("%H:%M:%S", time.gmtime(end - start))
	print('\nDuration:', duration)
	sys.stdout.close()
