# Transformer-couplet generator

#### Author: Zhang Kerui

#### Github: https://github.com/ISS-Kerui/Transformer-couplet-generator

## 1. Introduction

This project is the assignment of text processing course. The aim of this project is to implement a language **generation** model.

In this project, the task we selected is to automatically generate **Chinese couplets**.  Couplets are one of the traditional Chinese cultures, which are generally divided into the first sentence and the second sentence. The input of our model is the first sentence, and we hope to get the output of the next sentence. Here is one example:

First sentence：爆竹除旧岁

Second sentence：春暖入屠苏

Our model is divided into two parts. First, we use the latest general language model BERT for word embedded. Here we use the bert-as-service model provided by [hanxiao](https://github.com/hanxiao/bert-as-service).

Then we used a *Transformer* model to do encode and decode. 

The dataset we used is collected by [wb14123](https://github.com/wb14123). This dataset contains more than **700,000** couplets.

## 2. Getting started

##### Requriments:

1. Python 3.x
2. Pytorch 4.1+
3. tqdm
4. numpy
5. bert-as-service

**Step1:**

`git clone https://github.com/ISS-Kerui/Transformer-couplet-generator & cd Transformer-couplet-generator `

**Step2:**

`pip install -r requirements.txt`

**Step3:**

`python preprocess.py`

**Step4:**

`python train.py --cuda --pretrain_emb`  











