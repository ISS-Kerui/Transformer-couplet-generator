import argparse
import os
import pickle
import numpy as np  
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import TransformerSummarizer
from data.myLoader import LoadDataset, collate_fn, create_dict
description = 'Utility for sampling summarization.'

parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--inp', metavar='I', type=str, default='sample', help='name sample part of dataset')
parser.add_argument('--out', metavar='O', type=str, default='./generated.txt', help='output file')
parser.add_argument('--prefix', metavar='P', type=str, default='simple-summ', help='model prefix')
parser.add_argument('--dataset', metavar='D', type=str, default='./dataset', help='dataset folder')
parser.add_argument('--limit', metavar='L', type=int, default=30, help='generation limit')

args = parser.parse_args()

bpe_model_filename = os.path.join('./models_dumps', args.prefix, args.prefix + '_bpe.model')
model_filename = os.path.join('./models_dumps', args.prefix, 'RL_50_based130.model')
model_args_filename = os.path.join('./models_dumps', args.prefix, args.prefix + '.args')
emb_filename = os.path.join('./models_dumps', args.prefix, 'embedding.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dic = create_dict('./dataset/train.tsv')
test_loader = LoadDataset('./dataset/test.tsv',dic)
args_m = pickle.load(open(model_args_filename, 'rb'))

model = TransformerSummarizer(**args_m)

model.load_state_dict(torch.load(model_filename))
model.to(device)
model.eval()
epoch_size = test_loader.__len__()
per_epoch_iters = epoch_size // 100
custom_loader = DataLoader(test_loader, batch_size = 100, shuffle=False, \
collate_fn= collate_fn)
batch_iterator = iter(custom_loader)
progress_bar = tqdm(range(per_epoch_iters))
with torch.no_grad():
    summ = []
    summ_id = []
    truth = []
    truth_id = []
    for i in progress_bar:
        sources, summaries = next(batch_iterator)
        sources = np.array(list(sources))
        summaries = np.array(list(summaries))
        sources = torch.from_numpy(sources)
        sources = sources.to(device)
        summaries = torch.from_numpy(summaries)
        summaries = summaries.to(device)
        seq = model.sample(sources, args.limit)
        sum_words,sum_id = test_loader.decode(seq,dic.id2word)
        summ += sum_words
        summ_id += sum_id
        truth_words,ids = test_loader.decode(summaries,dic.id2word)
        truth += truth_words
        truth_id += ids
    with open(args.out, 'w') as f:
        f.write('\n'.join(summ_id))
    with open('summary.txt','w') as f:
        f.write('\n'.join(truth_id))

