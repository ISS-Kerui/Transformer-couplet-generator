import argparse
import logging
import os
import pickle
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.autograd import Variable
from tqdm import tqdm
#from data import DataLoader
from data.myLoader import LoadDataset, collate_fn, create_dict
from model import TransformerSummarizer
from torch.utils.data import DataLoader
description = 'NN model for abstractive summarization based on transformer model.'
epilog = 'Model training can be performed in two modes: with pretrained embeddings or train embeddings with model ' \
         'simultaneously. To choose mode use --pretrain_emb argument. ' \
         'Please, use deeper model configuration than this, if you want obtain good results.'

parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', metavar='DIR', type=str, default='./dataset', help='dataset directory')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda')
parser.add_argument('--vocab_size', metavar='V', type=int, default=4000, help='vocabulary size')
parser.add_argument('--pretrain_emb', action='store_true', help='use pretrained embeddings')
parser.add_argument('--emb_size', metavar='E', type=int, default=768, help='embedding size')
parser.add_argument('--model_dim', metavar='MD', type=int, default=768, help='dimension of the model')
parser.add_argument('--n_layers', metavar='NL', type=int, default=1, help='number of transformer layers')
parser.add_argument('--n_heads', metavar='NH', type=int, default=8, help='number of attention heads')
parser.add_argument('--inner_dim', metavar='ID', type=int, default=768, help='dimension of position-wise sublayer')
parser.add_argument('--dropout', metavar='D', type=float, default=0.1, help='dropout probability')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epoch', metavar='IT', type=int, default=10, help='number of epoches')
parser.add_argument('--train_bs', metavar='TR', type=int, default=64, help='train batch size')
parser.add_argument('--test_bs', metavar='TE', type=int, default=16, help='test batch size')
parser.add_argument('--test_interval', metavar='T', type=int, default=100, help='the test interval')
parser.add_argument('--sample_interval', metavar='S', type=int, default=100, help='the sample interval')
parser.add_argument('--train_interval', metavar='TL', type=int, default=100, help='train log interval')
parser.add_argument('--train_sample_interval', metavar='TS', type=int, default=500, help='train sample interval')
parser.add_argument('--log', metavar='L', type=str, default='./logs', help='logs directory')
parser.add_argument('--prefix', metavar='TP', type=str, default='gen', help='model prefix')
parser.add_argument('--saved_model', type=str, default=None, help='saved model name')
args = parser.parse_args()

# Make some preparations:

log_filename = os.path.join(args.log, args.prefix + '.log')
emb_filename = os.path.join('./models_dumps', args.prefix, 'embedded_words.npy')
dump_filename = os.path.join('./models_dumps', args.prefix, args.prefix + '.model')
args_filename = os.path.join('./models_dumps', args.prefix, args.prefix + '.args')
bpe_model_filename = os.path.join('./models_dumps', args.prefix, args.prefix + '_bpe.model')

for path in [log_filename, dump_filename]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Customize logs for printing
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ])

# Check CUDA and choose device
if torch.cuda.is_available():
    if  not args.cuda:
        logging.info('You have a CUDA device, so you should probably run with --cuda')
        
else:
    if  args.cuda:
        logging.warninig('You have no CUDA device. Start learning on CPU.')

device = torch.device("cuda"  if args.cuda and torch.cuda.is_available() else "cpu")
writer = SummaryWriter(os.path.join(args.log, args.prefix))
if args.pretrain_emb:
    embeddings = torch.from_numpy(np.load(emb_filename)).float()
    logging.info('Use vocabulary and embedding sizes from embedding dump.')
    vocab_size, emb_size = embeddings.shape
else:
    embeddings = None
    vocab_size, emb_size = args.vocab_size, args.emb_size

dic = create_dict('./dataset/train.tsv')
logging.info('Loading dataset')
#loader = DataLoader(args.dataset, ['train', 'test'], ['src', 'trg'], bpe_model_filename)
train_loader = LoadDataset('./dataset/train.tsv',dic)
test_loader = LoadDataset('./dataset/test.tsv',dic)
m_args = {'max_seq_len':50, 'vocab_size': vocab_size,
          'n_layers': args.n_layers, 'emb_size': emb_size, 'dim_m': args.model_dim, 'n_heads': args.n_heads,
          'dim_i': args.inner_dim, 'dropout': args.dropout, 'embedding_weights': embeddings}

m_args['embedding_weights'] = None

model = TransformerSummarizer(**m_args).cuda()
if args.saved_model != None:
    saved_model = os.path.join('./models_dumps', args.prefix, args.saved_model)
    state_dict = torch.load(saved_model)
    model.load_state_dict(state_dict)
    print ('saved model has been loaded...')

logging.info('Create model')



epoch_size = train_loader.__len__()
per_epoch_iters = epoch_size // args.train_bs

optimizer = Adam(model.learnable_parameters(), lr=args.learning_rate, amsgrad=True, betas=[0.9, 0.98], eps=1e-9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=6, verbose=True)
logging.info('Start training')
for i in range(args.epoch):
    try:
        custom_loader = DataLoader(train_loader,batch_size=args.train_bs, shuffle=True,collate_fn=collate_fn)
        batch_iterator = iter(custom_loader)
        progress_bar = tqdm(range(per_epoch_iters))
        loss_sum = 0.
        for i in progress_bar:
            sources, summaries = next(batch_iterator)
            sources = np.array([np.array(s) for s in sources])
            summaries = np.array([np.array(s) for s in summaries])
            sources = torch.from_numpy(sources)
            try:
                summaries = torch.from_numpy(summaries)
            except:
                print (summaries)
                continue
            #sources, summaries = Variable(sources,requires_grad=True), Variable(summaries)
            sources, summaries = sources.to(device), summaries.to(device)
            loss, seq = model.train_step([sources,summaries],optimizer)

            if i % args.train_interval == 0:
                logging.info('Iteration %d; Loss: %f', i, loss)
                writer.add_scalar('Loss', loss, i)

            if i % args.train_sample_interval == 0:
                texts, _ = train_loader.decode(sources,dic.id2word)
                originals, _ = train_loader.decode(summaries,dic.id2word)
                generateds, _ = train_loader.decode(seq,dic.id2word)
                text = texts[0]
                original = originals[0]
                generated = generateds[0]
                logging.info('Train sample:\nText: %s\nOriginal summary: %s\nGenerated summary: %s',
                         text, original, generated)

        
    
    except RuntimeError as e:
       logging.error(str(e))
       continue
    except KeyboardInterrupt:
       break

torch.save(model.cpu().state_dict(), dump_filename)
pickle.dump(m_args, open(args_filename, 'wb'))
logging.info('Model has been saved')
