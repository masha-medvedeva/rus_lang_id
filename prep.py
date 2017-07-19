import re, os, random
import glob
#pseudo-random, keep the number constant to be able to compare results
random.seed(132)

def extract_words_and_labels(file_name):
	print('extracting data and labels..')
	all_labels = []
	all_lines = []
	labels = []
	f = open(file_name, 'r')
	for line in f:
		line = re.sub('\n', '', line)
		line = line.split('\t') #split by the tab between the line and the language
		all_lines.append(line[0])
		all_labels.append(line[1])

	assert(len(all_labels)==len(all_lines))

	group_labels = []
	for l in all_labels:
		if l in ['tat', 'bak', 'chv', 'sah', 'tyv', 'alt', 'krc', 'nog', 'kjh', 'cjs', 'kum', 'kim']:
			group_labels.append('TRK')
		if l in ['che', 'inh', 'lez', 'ava', 'lbe', 'tab', 'udi', 'dar', 'tkr', 'kas', 'aqc', 'rut']:
			group_labels.append('ND')
		if l in ['mhr', 'myv', 'kpv', 'udm', 'mdf', 'koi', 'yrk', 'yux', 'mns', 'kca', 'ykg']:
			group_labels.append('UR')
		if l in ['kbd', 'ady', 'abq']:
			group_labels.append('AA')
		if l in ['evn', 'gld', 'eve']:
			group_labels.append('TNG')
		if l in ['bxr', 'xal']:
			group_labels.append('MNG')
		if l in ['ckt', 'kpy']:
			group_labels.append('CK')
		if l in ['rus', 'ttt']:
			group_labels.append('IE')
		if l == 'niv':
			group_labels.append('ISO')


	return all_lines, group_labels, all_labels
	#return all_lines[:5000], group_labels[:5000], all_labels[:5000]
	#try out on smaller dataset, 5000 is the length of test data
