# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:41:36 2019

Runs model pipeline
- define parameters
- structure data
- prepare sequence data type
- run model 

@author: Austin Bell
"""

from torchtext import data
import sys, os
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter

basePath = "C:/Users/Austin Bell/Documents/NLP/NER"
sys.path.append(basePath)
os.chdir(basePath)
from Utils import *
from Train import *
from Model import *


##### Define data parameters
trn_file = "engtrain.bio"
test_file = "engtest.bio"

#trn_file = "GeneTrain"
#test_file = "GeneTest"


emb_vectors = "glove.6B.100d" # Utilize Glove embeddings (100 dim)
min_word_freq = 0 # setting to 0 because gene names are probably extremely rare and we do not want to <unk> these



############################
# Model Parameters
############################
bs = 64 # Batch size
num_layers = 1 # number of layers
embedding_sz = 100 # embedding length
hidden_sz = 200 
char_embedding_sz = 50
char_hidden_sz = 100
lr = .001
epochs = 100
dropout = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################
# Text Processing
############################

# Define Fields
TEXT = data.Field(sequential = True, pad_first = True, batch_first = True)
TAGS = data.Field(sequential = True, pad_first = True, batch_first = True)
    

# split data into inputs and 
train, test = sequenceSplits(TEXT, TAGS, trn_file, test_file)

# Build vocab objects for mapping idx to tokens
TEXT.build_vocab(train, 
                 min_freq = min_word_freq,
                 vectors=emb_vectors)

vocab = TEXT.vocab

# Build the mapped tag vocab object
TAGS.build_vocab(train)
tagVoc = TAGS.vocab


# create data loaders
trn_iter = constructIters(bs, train)
test_iter = constructIters(bs, test, reorder = False)
# Check that this is the correct way to shuffle data

# Create character map
char_map = charMap(trn_file)
char_vocab_sz = len(char_map)

############################
# Run Model
############################

vocab_sz = len(vocab)
num_tags = len(tagVoc)
vectors = TEXT.vocab.vectors

model = biLSTM(vocab_sz, embedding_sz, hidden_sz, num_tags, 
                     vectors, num_layers, bs, dropout,
                     char_vocab_sz, char_embedding_sz, char_hidden_sz).to(device)

# define the optimizer and loss function
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
criterion = nn.NLLLoss().to(device).to(device)

# Train algorithm
num_batches = len(trn_iter)
losses = []

for i in range(epochs):
    model.train()
    total_loss = 0
    running_correct = 0
    running_total = 0
    for j, batch in enumerate(trn_iter):

        total_loss, running_total, running_correct = trainModel(model, batch, total_loss,
                                                                running_total, running_correct,
                                                                criterion, optimizer, device,
                                                                char_map, vocab)
        
        if j % 75 == 0:
            print("Epoch: {} out of: {}".format(i+1, epochs))
            print("Batch number: {} out of: {}".format(j, num_batches))
            print("Accuracy: ", (int(running_correct)/int(running_total)))
            
    print("Epoch Loss: ", total_loss)
    losses.append(total_loss)
    
print("Total Training Losses: ", losses)
print("Training Accuracy: ", (int(running_correct)/int(running_total)))

# Evaluate the algorithm
with torch.no_grad():
    correct = 0
    total = 0
    
    # Initialize evaluator    
    evaluation = evaluator(tagVoc.itos)
    
    
    for xy in test_iter:
        sentences, tags = xy.Inputs, xy.Tags.to(device)
        sentences = sentences.to(device)
        
        # forward pass
        outputs = model.forward(sentences, char_map, vocab, device)
        values, indices = torch.max(outputs,2) # max on tag dim
        
        # evaluate total accuracy and by tag
        total += indices.size(0)*indices.size(1)
        correct += int(torch.sum((indices == tags)))
        
        evaluation.evalByTag(indices, tags, tagVoc)

    print("Final Accuracy: ", correct/total)
    for t in evaluation.tag_names[2:]: # skip <unk> and <pad>
        try:
            print("{} tag accuracy: {}".format(t, evaluation.tag_counts[t].get_accuracy()))
            print("{} tag precision: {}".format(t, evaluation.tag_counts[t].get_precision()))
            print("{} tag recall: {}".format(t, evaluation.tag_counts[t].get_recall()))
            print("{} tag F score: {}".format(t, evaluation.tag_counts[t].get_fscore()))
        except:
            print("0 {}".format(t))
        
    
        
    
    
    