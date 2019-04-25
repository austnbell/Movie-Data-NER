# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:03:34 2019

Prep data loaders
get tags

@author: abell
"""
from torchtext import data
from torchtext import datasets
import torch 
from collections import Counter

##### Torchtext pre-processing
def sequenceSplits(TEXT, TAGS, trn_file, test_file):
    # Split by fields for train, val and test
    train, test = datasets.SequenceTaggingDataset.splits(
            path="./Data/Inter/",
            fields = (('Tags', TAGS), ('Inputs', TEXT)),
            train = trn_file, test=test_file,
            separator="\t"
            )
    
    return train, test

def constructIters(bs, file, reorder = True):
        
    # We use bucket Iterator so that we can group the batches to reduce the amount of necessary padding
    # similarly, we want to ensure that each batch contains a wide variety of information
    
    if reorder:
        reiterFunc = data.BucketIterator
    else:
        reiterFunc = data.Iterator
    
    iterator = reiterFunc(
            (file),
            batch_size =(bs),
            sort_within_batch = False)
    
    return iterator

# reads the words in a file and maps characters to indices
def charMap(file_name, min_char_freq=1):
    
    file = open("./Data/Inter/%s" % file_name).readlines()
    
    char_freq = Counter()
    
    for line in file:
        if line != "\n":
            word = line.split("\t")[1]
            char_freq.update(list(word))
        
    # convert frequency count to map if it is greater than min_char_freq
    char_map = {char:idx+2 for idx, char in enumerate(char_freq) if char_freq[char] >= min_char_freq}
    char_map["<pad>"] = 1 # add padding token
    char_map["<unk>"] = 0 # add unknown token
    
    return char_map

# takes in a single column from a batch of sentences and converts to a batch of characters
# the character LSTM will then take in column by column
# we extract the last vector of the forward and backward char lstm and concatenate both vectors to the relevant word vectors
def charBatch(batch_col, char_map, vocab):
    words = [vocab.itos[int(word)] for word in batch_col] # convert indices to words
    
    global chars, c_lengths
    # convert words to character indices
    chars = list(map(lambda word: [char_map[char] if word not in ["<pad>", "<unk>"] else char_map[word] for char in word], words))
    chars = list(map(lambda char: [1] if char ==  [1, 1, 1, 1, 1] else char, chars)) # convert padding
    chars = list(map(lambda char: [0] if char ==  [0, 0, 0, 0, 0] else char, chars)) # convert unknown
    
    # pad character lists
    max_length = len(max(chars, key = len)) # padding length
    padded_chars = list(map(lambda char: [char_map['<pad>']] * (max_length - len(char)) + char, chars))
    
    padded_chars = torch.LongTensor(padded_chars) # convert to tensor
    
    return padded_chars
    


