# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:11:22 2019

Defines BiLSTM Model 

@author: Austin Bell
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Utils import *

# Character level Bi-LSTM
class charLSTM(nn.Module):
    def __init__(self, char_vocab_sz, char_embedding_sz, char_hidden_sz, bs):
        super(charLSTM, self).__init__()
        self.hidden_sz = char_hidden_sz
        self.embedding_sz = char_embedding_sz
        self.vocab_sz = char_vocab_sz
        self.bs = bs
        
        self.char_embedding = nn.Embedding(char_vocab_sz, char_embedding_sz, padding_idx = 1)
        
        self.char_lstm = nn.LSTM(char_embedding_sz,
                                 char_hidden_sz // 2,
                                 1,
                                 batch_first = True,
                                 bidirectional = True) # finish at LSTM extract results

    def forward(self, chars):
        # add embedding layer
        char_embs = self.char_embedding(chars)
        
        # LSTM layer
        char_lstm_out, char_hs = self.char_lstm(char_embs)
        
        # select the nth cell of of the forward pass
        # select the 1st cell of the backward pass
        # these are the last hidden states of each cell 
        last_state = torch.cat([char_lstm_out[:,-1,:self.hidden_sz // 2],
                                char_lstm_out[:,-1,self.hidden_sz // 2:]], dim = 1) # (bs, cell, hidden units)
        
        return last_state.view(last_state.size(0), 1, last_state.size(1)) 
        

# Bring it all together 
# Runs a Bi-LSTM for each sentence with character LSTM occurring at every word
class biLSTM(nn.Module):
    def __init__(self, vocab_sz, embedding_sz, hidden_sz, num_tags, 
                 vectors, num_layers, bs, dropout,
                 char_vocab_sz, char_embedding_sz, char_hidden_sz):
        super(biLSTM, self).__init__()
        self.hidden_sz = hidden_sz
        self.num_tags = num_tags
        self.num_layers = num_layers
        self.bs = bs
        
        # Initialize the embedding layer using pre-trained vectors
        self.embedding = nn.Embedding(vocab_sz, embedding_sz, padding_idx = 1).from_pretrained(vectors, freeze = False) 
        
        # Bi-directional char LSTM
        self.char_lstm = charLSTM(char_vocab_sz, char_embedding_sz, char_hidden_sz, bs)
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(embedding_sz + char_hidden_sz, 
                            hidden_sz // 2,
                            num_layers, 
                            dropout = dropout,
                            batch_first = True,
                            bidirectional = True)
        
        # Linear output layer
        self.hidden2tag = nn.Linear(hidden_sz, num_tags) # *2 for bi lstm    
        
    
    def forward(self, inputs, char_map, vocab, device):

        # Add embedding layer
        embs = self.embedding(inputs)
        
        # Run Character LSTM
        seq_len = inputs.size(1)
        full_cLSTM = []
        for i in range(seq_len):
            col = inputs[:,i] # selects a single batch column
        
            char_col = charBatch(col, char_map, vocab) # convert batch column to character batch
            char_hs = self.char_lstm(char_col.to(device))
            full_cLSTM.append(char_hs)
            
        # Concatenate tensors
        full_cLSTM = torch.cat(full_cLSTM, dim = 1) # cat by sequence length (bs, seq_len, hidden_units)
        emb_cLSTM = torch.cat([embs, full_cLSTM], dim = 2) # cat embedding layer + hidden units (bs, seq_len, hs+emb)

        
        # Forward propagation
        lstm_out, _ = self.lstm(emb_cLSTM)
        
        # decode output and push through softmax
        out = self.hidden2tag(lstm_out)
        scores = F.log_softmax(out, dim = 2)
        
        return scores