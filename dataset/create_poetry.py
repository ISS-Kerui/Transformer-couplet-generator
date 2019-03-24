#Create poetry dataset
import re
def remove_punc(string):
	punc = '[,.!;\']'
	return re.sub(punc, '', string)
txt_path = 'robert_frost.txt'
sources = []
targets = []
with open(txt_path,'r') as f:
	lines = f.readlines()
	for i in range(len(lines)-1):
		if lines[i] !='\n' and lines[i+1] != '\n':
			sources.append(remove_punc(lines[i].strip()))
			targets.append(remove_punc(lines[i+1].strip()))

assert len(sources) == len(targets)
print (len(sources))
with open('poetry_train.tsv','w') as f:
	for i in range(len(sources[:1100])):
		f.write(sources[i]+'\t'+targets[i]+'\n')

with open('poetry_test.tsv','w') as f:
	for i in range(len(sources[1100:])):
		f.write(sources[i]+'\t'+targets[i]+'\n')
