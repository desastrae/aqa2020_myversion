#!/bin/python

# Import nn models

# To take relative import path from current directory
import sys
sys.path.append('./')

from models.modeling import WordEmbedding

import torch
from torch import nn
from torch import optim
import time
import re
import argparse
from tqdm import tqdm, trange
import numpy as np
import pickle
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.switch_backend('agg')

'''
The goal of this model/training script is to obtain better Word Embeddings
'''

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


def tok(sent):
    ''' This is how sample tokenization might look like; you can come up with your own tokenizer'''

    # print(sent)
    # sent = re.sub(r'([a-zA-Z0-9])([?!.]) ([A-Z])',r'\1 \2 \3',sent) #new sentence
    # sent = re.sub(r' ([?!.]) ([A-Z])',lambda m: m.group(0).lower(),sent) # lower case the word after newline
    # sent = re.sub(r'([a-zA-Z0-9]+)([.-])([A-Za-z0-9]+)',r'\1 \2 \3',sent) #compound splitting
    # sent = re.sub(r'([a-zA-Z0-9])([,;:-]) ([a-z])',r'\1 \2 \3',sent) #commas, semi-columns etc are to be sepearted

    sent = re.sub(r'(["()\[\]])', r' \1 ', sent)  # Ensure ",(,),[,] are always separate
    sent = re.sub(r'([.!?:-])+', r'\1', sent)  # merge multiple punctuations to single punctuation
    sent = re.sub(r'([a-zA-Z0-9])([?!.:;\-,]) ', r'\1 \2 ', sent) # if you find any where "alphabets[punctuation]" - separate punctuation
    sent = re.sub(r'([a-zA-Z0-9])([.?!,:]+)$', r'\1 \2', sent)  # last line
    sent = re.sub(r' +', r' ', sent)  # merge all spaces to 1 space
    sent = sent.lower()

    return sent


def make_pretrained_embeddmatrices(wvecfile,idx2word,vocab_size,wvec_size):
    wvecdict={}
    with open(wvecfile, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")
            vec = fields[1:]
            word = fields[0]
            wvecdict[word] = np.array([np.float64(w) for w in vec])
    npmat = np.matrix(np.zeros((vocab_size, wvec_size)))
    for i in range(vocab_size):
        if idx2word[i] in wvecdict.keys():
            npmat[i] = np.array(wvecdict[idx2word[i]])
        else:
            npmat[i] = np.random.rand(wvec_size)
            np.array(wvecdict[idx2word[i]])  # what's that?!  CHANGED

    return npmat


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, input_ids):

        self.unique_id = unique_id
        self.input_ids = input_ids


def read_vocab(input_file):
    count = 0
    vocabcount = dict()
    vocab = dict()
    threshold = 3  # The word should occur at least 3 times to be considered
    pad_token = 0
    unk_token = 1
    vocab["#pad"] = 0
    vocab["#unk"] = 1
    vocab["#rep"] = 2

    with open(input_file, "r", encoding='utf-8') as reader:
        for line in reader:
            line = tok(line.strip())

            for word in line.split(" "):
                if word in vocabcount:
                    vocabcount[word] += 1
                else:
                    vocabcount[word] = 1

    ind = 3

    for word in vocabcount:
        if vocabcount[word] <= threshold:  # Rare words are ignored!
            vocab[word] = unk_token           #  CHANGED
            # pass
        else:
            vocab[word] = ind
            ind += 1
    return vocab, ind


def read_textfile_as_context_windows_n(input_file, vocab, window=5):

    ''' Takes a text and create n grams  and then convert into features'''
    ''' features is nothing but a python object with the data you want to read per batch '''

    # if features is already stored in a file, load it
    if os.path.isfile(input_file+".features.pickle"):
        print("pickled input features from", input_file, "exist")
        features = pickle.load(open(input_file+".features.pickle", "rb"))
        return features

    text1_lines=[]
    features = []
    pad_token = vocab["#pad"]
    unk_token = vocab["#unk"]

    count = 0
    all_ngrams = []
    with open(input_file, "r", encoding='utf-8') as reader:
        for line in reader:

            line = tok(line.strip())
            tokens = [vocab["#rep"]]+[token for token in line.split(" ") if token != "" ]
            ''' Here we are adding prefix a Beginning of sentence, 
            this helps to learn meaning for the words that are in the beginning '''

            ngrams = zip(*[tokens[i:] for i in range(window)])
            all_ngrams += ngrams
             
    for l in range(len(all_ngrams)):
        idline = [vocab[word] if word in vocab else unk_token for word in all_ngrams[l]]
        features.append(InputFeatures(unique_id=l, input_ids=idline, ))

    # Store the features in a file for future reading
    pickle_out = open(input_file+".features.pickle", "wb")
    pickle.dump(features, pickle_out)
    pickle_out.close()

    return features        


def tsne_plot(model, words, vocab, filename):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in words:
        i = vocab[word]
        vec = [it.item() for it in model.WE_layer.weight.data[i]]  # This line should be changed according to your model
        tokens.append(vec)
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    fig = plt.figure()
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    # plt.show()
    fig.savefig(filename)


def plot(model, vocab, filename):
    words = []
    with open("data/plot_words.txt", "r") as f:
        for line in f:
            line = line.strip()
            words.append(line)
    tsne_plot(model, words, vocab, filename)


def evaluate(model, test_dataloader):
 
    model.eval()
    eval_loss=0
    eval_total=0
    for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration", disable=args.gpu_number not in [-1, 0])):

        ''' DATA '''
        batch = [t.to(device) if t is not None else None for t in batch]  # multi-gpu does scattering it-self

        ''' Forward Pass '''
        log_probs = model(input_ids)
        loss = -log_probs.sum(1).sum(0)  # sum all the losses
        eval_loss += loss.item()
        eval_total += input_ids.size(0)*(input_ids.size(1)-1)  # total number of predictions

    return 2 ** (eval_loss/ eval_total)  # Perplexity


# Main ###

# Tip : The arguments of the python script can be neatly passed by using argparse library "
parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--train_file", default=None, type=str, required=True,
                       help="train file that you want to read")
parser.add_argument("--dev_file", default=None, type=str, required=True,
                       help="dev file that you want to read")
parser.add_argument("--load_model", default=None, type=str, required=False,
                       help="path for the trained model to load")
parser.add_argument("--embedd_file", default=None, type=str, required=False,
                       help="path for the pretrained word embeddings file")

parser.add_argument("--train_batch_size", default=128, type=int, help="Train batch size for training.")
parser.add_argument("--test_batch_size", default=128, type=int, help="Valid batch size for training.")
parser.add_argument("--num_train_epochs", default=6, type=int, help="Number of training epochs"
                                                                    "[The loss function will be minimized "
                                                                    "for epoch number of rounds on training data]")
parser.add_argument("--wemb_size", default=142, type=int, help="Word Embedding size")
parser.add_argument("--hidden_size", default=512, type=int, help="hidden size for transformers encoder")
parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
parser.add_argument("--gpu_number",
                        type=int,
                        default=-1,
                        help="gpu_number=-1 for distributed training on gpus;"
                             "gpu_number=1 makes the code run on gpu 1; This is also call \"local_rank\"")
parser.add_argument("--window_size",
                        type=int,
                        default=5,
                        help=" the ngram size that you want to use as context ")
parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

args = parser.parse_args()

# Read Vocab
# if a vocab file already exists, just load it
if os.path.isfile(args.train_file+".vocab.pickle"):
    vocab = pickle.load(open(args.train_file+".vocab.pickle", "rb"))
    vocab_size = len(vocab.keys())
else:
    vocab, vocab_size = read_vocab(args.train_file)
    pickle_out = open(args.train_file+".vocab.pickle", "wb")
    pickle.dump(vocab, pickle_out)
    pickle_out.close()

    vocabf = open("data/vocab.txt","w")
    for key, value in vocab.items():
        vocabf.write(str(key)+"\t"+str(value)+"\n")
    vocabf.close()

pad_token = vocab["#pad"]
idx2word=dict((v, k) for k, v in vocab.items())

print("Vocab size", vocab_size)


model = WordEmbedding(vocab_size=vocab_size, isize=args.wemb_size, hsize=args.hidden_size)


opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# model should be in train mode
model.train()
start = time.process_time()

# Read data


# Read DATA into tensors; Use DataLoader to iterate in batches; 
# Following libraries are used
# TensorDataset
# DataLoader
# RandomSampler

train_features = read_textfile_as_context_windows_n(args.train_file, vocab, window=args.window_size)
print("Read training file", args.train_file)
test_features = read_textfile_as_context_windows_n(args.dev_file, vocab, window=args.window_size)
print("Read test file", args.dev_file)

print("Training size, N:", len(train_features))
print("Dev size: ", len(test_features))

'''..So far we read words into numbers and create list of objects called features '''

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids)
'''..So far, we made a list of tensors for all the training data. After the above steps, the training data lies on CPU; 
  If the training data is too big, you should split it and iterate over each split with the above commands'''

train_sampler = RandomSampler(train_data)
'''..RandomSampler randomly samples from the data '''

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
'''..Data Loader for training set takes in the tensor and iterate over training data each tensor 
with batch_size as specified'''

all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
test_data = TensorDataset(all_input_ids)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)
'''.. do the same for test/dev file '''

start = time.process_time()

# The below assigning of device is copied from huggingface github code (pytorch transformers)

# if you want distributed computing on multiple gpus / if you want to run on cpu
if args.gpu_number == -1 or args.no_cuda:
    device = torch.device(
        # picks cuda if there exists gpu in your machine; #run-this-command
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()  # run-this-command
else:
    torch.cuda.set_device(args.gpu_number)
    device = torch.device("cuda", args.gpu_number)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend='nccl')
''' ..Get device: this decides where we will be running our code : CPU / GPU, if GPU multiple gpus or one gpu'''


''' Using Pretrained Embeddings '''
if args.load_model is None and args.embedd_file is not None:
    wvec_size = model.WE_layer.weight.size()[1]
    pretrained_wemb_matrix = make_pretrained_embeddmatrices(args.embedd_file, idx2word, vocab_size, wvec_size)
    model.WE_layer.load_state_dict({'weight': torch.from_numpy(pretrained_wemb_matrix).float()})

if args.load_model is not None:
    model = torch.load(args.load_file)
   
model.to(device)

'''Check before training how well the embedding model is '''
plot(model, vocab, "data/WE_beginning.png")


best_eval_loss = 0
# Training
for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
    tr_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.gpu_number not in [-1, 0])):
        ''' Zero the gradients, otherwise gradients accumulate '''
        opt.zero_grad()

        ''' DATA '''
        batch = [
               t.to(device) if t is not None else None for t in batch]  # multi-gpu does scattering it-self
        input_ids = batch[0]

        ''' Forward Pass '''
        log_probs = model(input_ids)

        loss = -(log_probs.sum(-1).mean(0))
        # minimizing - log(prob) => maximizing probabilities; mean(0) - averages loss over batch_size

        # debug:
        # print(loss)     # CHANGED
        # Note that loss should always be positive: probs are in range (0,1);
        # log(probs) are in range (-inf, 0); -log(probs) are in range (0,inf)
        # input("loss")

        tr_loss += loss

        ''' Backward Pass '''
        loss.backward()

        ''' Optimization step '''
        opt.step()

        '''After 1000 steps, check for evaluation scores
         if the training set is too big, you can evaluate and save at certain number of steps instead'''

        if step % 5000 == 0 and step != 0:  # evaluate, save at 30k steps?
            eval_loss = evaluate(model, test_dataloader)
            print("At step", step, ", Avg training loss=", tr_loss.item()/step, "Evaluation loss=", eval_loss)
            if best_eval_loss <= eval_loss:
                plot(model, vocab, "data/WE_"+str(step)+"_steps.png")
                best_eval_loss = eval_loss
                torch.save(model, "saved_models/wordembedding.best.pt")

print("Training time ", time.process_time() - start)

