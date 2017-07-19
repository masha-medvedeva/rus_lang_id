import glob, re
from nltk.tokenize import sent_tokenize

data_en_folder = glob.glob(argv[1]+'/*.json')

d = {}
texts = []
labels = []
domains = []

for file_name in data_en_folder:
	f = open(file_name, 'r')
	print(file_name)
	text_lang = []
	for line in f:
		#extract language
		lang = re.search('    "language": "(.*)_url_lists"', line)
		if lang != None:
			language = lang.group(1)
		text = re.search('"text": "(.*)"', line)
		#extract text
		if text != None:
			text_split = sent_tokenize(text.group(1))
			text_split = [i[:300] for i in text_split]
			text_lang.extend(text_split)

		#extract domains
		dom = re.search('"url": "(.*)"', line)
		if dom != None:
			domain = dom.group(1)

	texts.extend(text_lang)
	labels.extend([language]*len(text_lang))
	domains.extend([domain]*len(text_lang))
	assert(len(labels) == len(texts))


for i in range(len(texts)):
	d[texts[i]] = labels[i] + '\t' + domains[i]

rus_unigrams = open('unigrams.cyr.lc', 'r')
unigrams = []
for line in rus_unigrams:
	unigram = line.split('\t')[0]
	unigrams.append(unigram)

extracted = open('extracted.txt', 'w')

print('writting..')
count = 0
for i in d:
	#print(i.decode('cp1252', 'replace'))
	i2 = set(i.lower())

	if (len(i2 & set(['ø', 'õ', 'ó', 'à', 'ã', 'ð', 'å', 'â', 'ñ' 'þ', 'á', 'ä', 'ç'])) == 0) \
			and len(i2 - set('qwertyuiopasdfghjklzxcvbnm.,\\!?1234567890/#:;[] ()_«»-=~`')) > 0 \
			and len(i2 & set('ﺍﺏﺕﺙﺝﺡﺥﻩﻉﻍﻑﻕﺹṣﺽﺩﺫﻁﻙﻡﻥﻝﻱﺱﺵﻅﺯﻭﺭ')) == 0:
		num_rus = 0
		for word in i.split():
			if word.lower().strip(".;:!?/\\,#@$&)(\"") in unigrams:
				num_rus +=1
		if len(i.split())-num_rus > num_rus:
			if len(i.split()) > 2:
				extracted.write(i+'\t'+d[i]+'\n')
		elif float(len(i.split())) * 2 / 3 < num_rus:
			if len(i.split()) > 2:
				d[i] = 'rus' + d[i][3:]
				extracted.write(i+'\t'+d[i]+'\n')
				count += 1


print('Lines eliminated due to overwhelming amount of Russian words:', count)

l = set(labels)
print('TOKENS\n')
print('Languages:' + str(len(l)), l)
for label in l:
	print(label)
	count = 0
	extracted_count = open('extracted.txt', 'r')
	for line in extracted_count:
		line = re.sub('\n', '', line)
		line = line.split('\t')
		if label == line[-2]:
			count += len(line[0].split(' '))
	print(label + ' & ', str(count), '\\\\')


print('\n\n\nLINES\nLanguages:' + str(len(l)), l)
for label in l:
	count = 0
	extracted_count = open('extracted.txt', 'r')
	for line in extracted_count:
		line = re.sub('\n', '', line)
		line = line.split('\t')
		if label == line[-2]:
			count += 1
	print(label + ' & ', str(count), '\\\\')
