import argparse
import logging
import os
from data.myLoader import  create_dict
from data.utils import SequentialSentenceLoader, export_embeddings
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

description = 'Preprocessor for summarization dataset. Include unsupervised text tokenization and vector word' \
              'representation using word2vec model.' \
              'Dataset must consists of two parts: train and test stored in `train.tsv` and `test.tsv` respectively. ' \
              'After training embedding model is saving into dataset directory.'

parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='DIR', type=str, default='./dataset', help='dataset directory')
parser.add_argument('--vocab_size', metavar='V', type=int, default=4000, help='vocabulary size')
parser.add_argument('--emb_size', metavar='E', type=int, default=250, help='embedding size')
parser.add_argument('--workers', metavar='WS', type=int, default=4, help='number of cpu cores, uses for training')
parser.add_argument('--sg', action='store_true', help='use skip-gram for training word2vec')
parser.add_argument('--prefix', metavar='P', type=str, default='gen', help='model prefix')

args = parser.parse_args()

train_filename = os.path.join(args.dataset, 'train.tsv')
test_filename = os.path.join(args.dataset, 'test.tsv')
sp_model_prefix = os.path.join('./models_dumps', args.prefix, args.prefix + '_bpe')
sp_model_filename = sp_model_prefix + '.model'
embeddings_filename = os.path.join('./models_dumps', args.prefix, 'embedded_words.npy')


#Next , using BERT embeddings:
from bert_serving.client import BertClient
dic = create_dict('./dataset/train.tsv')
words = dic.word2id.keys()
bc = BertClient()
encoded_words = []
for word in words:
	encoded_word = bc.encode([word])
	encoded_words.append(encoded_word)
encoded_words = np.array(encoded_words).squeeze()
np.save(embeddings_filename,encoded_words)




# Export embeddings into lookup table:
#export_embeddings(embeddings_filename, sp, w2v_model)
#logging.info('Embeddings have been saved into {}'.format(embeddings_filename))
