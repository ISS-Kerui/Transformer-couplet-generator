from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import numpy as np
class MyDict():
	def __init__(self, texts):
		self.texts = texts
		self.word2id = {}
		self.id2word = {}
		self.id_num = 4
		self._word2id()
		self._id2word()
		
	def _word2id(self):
		
		self.word2id['PAD'] = 0
		self.word2id['UNK'] = 1
		self.word2id['s'] = 2
		self.word2id['/s'] = 3
		for text in self.texts:
			for word in text:
				if self.word2id.get(word) == None and word != ' ':
					self.word2id[word] = self.id_num
					self.id_num += 1

					

	def _id2word(self):
		self.id2word = {value:key for key,value in self.word2id.items()}
	def __len__ (self):
		return len(self.word2id.keys())
class LoadDataset(Dataset):
	def __init__(self, data_path,ditionary,img_transform = None):
		# part: train or test
		# data_path :train.tsv or test.tsv
		self.data_path = data_path
		self.ditionary = ditionary
		self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx = range(4)
		self.data_part = list(self.from_tsv(data_path)) 
		self.data_part = [self.pad(self.data_part[0],'source'),self.pad(self.data_part[1],'summary')]
		
		self.img_transform = img_transform
	def __getitem__(self,index):
		source_txt = self.data_part[0][index]
		target_txt = self.data_part[1][index]
		return ([source_txt, target_txt])
	def __len__(self):
		return len(self.data_part[1])
	def pad(self, data,tp):
		"""Add <sos>, <eos> tags and pad sequences from batch

		Args:
			data (list[list[int]]): token indexes

		Returns:
			list[list[int]]: padded list of sizes (batch, max_seq_len + 2)
		"""
		data = list(map(lambda x: [self.sos_idx] + x + [self.eos_idx], data))
		lens = [len(s) for s in data]
		if tp == 'source':
			max_len = 140
		elif tp == 'summary':
			max_len = 32
		for i, length in enumerate(lens):
			to_add = max_len - length
			data[i] += [self.pad_idx] * to_add
		return np.array(data)

	def from_tsv(self, part):
		"""Read and tokenize data from TSV file.

			Args:
				part (str): the name of the part.
			Yields:
				(list[int], list[int]): pairs for each example in dataset.

		"""
		
		with open(self.data_path) as file:
                        reader = csv.reader(file, delimiter='\t')
                        source_ids = []
                        target_ids = []
                        for row in reader:
                                try:
                                        #pdb.set_trace()
                                        source = row[0]
                                        target = row[1]
                                except:
                                        print (1)
                                        continue
                                source_id = []
                                target_id = []
                                for word in source.split(' '):
                                        single_word_id = self.ditionary.word2id.get(word)
                                        if single_word_id != None:
                                                source_id.append(single_word_id)
                                        else:
                                                source_id.append(self.ditionary.word2id.get('UNK'))
                                for word in target.split(' '):
                                        single_word_id = self.ditionary.word2id.get(word)
                                        if single_word_id != None:
                                                target_id.append(single_word_id)
                                        else:
                                                target_id.append(self.ditionary.word2id.get('UNK'))
                                source_ids.append(source_id)
                                target_ids.append(target_id)


                        return tuple([np.array(source_ids),np.array(target_ids)])
	def decode(self, outputs,id2word_dic):
            decoded_sentences = []
            decoded_ids = []
            outputs = outputs.cpu().numpy()
            for output in outputs:
                decoded_sentence = ''
                decoded_id = []
                for token in output:
                    if token == self.eos_idx:
                        break
                    decoded_id.append(token)
                    decoded_sentence += id2word_dic.get(token)
                decoded_sentences.append(decoded_sentence)
                decoded_id = [str(each_id) for each_id in decoded_id]
                decoded_id = decoded_id[1:-1]
                decoded_ids.append(' '.join(decoded_id))
            return decoded_sentences,decoded_ids

def collate_fn(batch):
	batch.sort(key=lambda x: len(x[1]), reverse=True)
	source, summary = zip(*batch)
	
	return source, summary

def create_dict(data_path):
        texts = []
        with open(data_path) as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                        try:
			#pdb.set_trace()
                                texts.append(row[0])
                                texts.append(row[1])
                        except:
                                print (row)
	#pdb.set_trace()
        frequency = defaultdict(int)
        for text in texts: 
                for token in text.split():
                        frequency[token] += 1 
        texts = [[token for token in text.split() if frequency[token] > 10] for text in texts]
        dic = MyDict(texts)
        print (len(dic))
        return dic

from torch.utils.data import DataLoader
import pdb
from collections import defaultdict
from bert_serving.client import BertClient
if __name__ == '__main__':
	data_path = 'train.tsv'
	dic = create_dict(data_path)
	# words = dic.word2id.keys()
	# bc = BertClient(ip='10.217.129.216')
	# encoded_words = []
	# for word in words:
	# 	print (word)
	# 	encoded_word = bc.encode([word])
	# 	encoded_words.append(encoded_word)
	# encoded_words = np.array(encoded_words).squeeze()
	# np.save('embedded_words.npy',encoded_words)
	dataset = LoadDataset('train.tsv',dic)
	# print (dataset.__len__())
	custom_loader = DataLoader(dataset,batch_size=16, shuffle=True,collate_fn=collate_fn)
	batch_iterator = iter(custom_loader)
	source, summary = next(batch_iterator)
	print (np.array(list(summary)).shape)

	#print (source)


