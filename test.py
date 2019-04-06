import numpy as np  
import argparse
from data.myLoader import create_dict
parser = argparse.ArgumentParser(description='')
parser.add_argument('--id_gen_txt', metavar='I_gen', type=str, default='id_generated_txt/generated.txt', help='input file')
parser.add_argument('--id_gold_txt', metavar='I_gold', type=str, default='id_generated_txt/summary.txt', help='input file')
parser.add_argument('--word_txt', metavar='O', type=str, default='./id_generated_word.txt', help='output file')
args = parser.parse_args()

gen_txt = args.id_gen_txt
gold_txt = args.id_gold_txt
out_txt = args.word_txt

def ids2sentence(ids ,dic):
    id2word = dic.id2word
    sentences = []
    for line in ids:
        sentence = []
        for _id in line.split():
            try:
                sentence.append(id2word[int(_id)])
            except:
                continue

        sentence =  "".join(sentence)
        sentences.append(sentence)
    return sentences



dic = create_dict('./dataset/train.tsv')
with open(gen_txt,'r') as f1:
    with open(gold_txt,'r') as f2:
        gen_ids = f1.readlines()
        gold_ids = f2.readlines()
        
        gen_sentence = ids2sentence(gen_ids,dic)
        gold_sentence = ids2sentence(gold_ids,dic)

with open(out_txt,'w') as f:
    for i in range(len(gen_sentence)):
        f.write('model_out: '+gen_sentence[i]+'\t'+'gold sentence: '+gold_sentence[i])
        f.write('\n')






