# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:41:36 2019

Training and eval functions

@author: Austin Bell
"""

from torch.autograd import Variable
import torch

def trainModel(model, batch, total_loss, running_total, running_correct, criterion, optimizer, 
               device, char_map, vocab):
    sentences, tags = Variable(batch.Inputs).to(device), Variable(batch.Tags).to(device)
    #batch_lengths = list(map(lambda char: [1 for char in chars if char != 1], sentences))
    
    # reset optimizer
    model.zero_grad()
    
    # forward pass
    outputs = model(sentences, char_map, vocab, device)
    _, preds = torch.max(outputs, 2) # I need to remove pads prior to prediction
    
    # Compute Loss
    outputs = outputs.transpose(1,2) # dim(batch, classes, words)
    loss = criterion(outputs, tags)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    running_correct += torch.sum((preds == tags).data)
    running_total += torch.sum((tags == tags).data)
    
    return total_loss, running_total, running_correct 


class neTypeCounts(object):
    """
    Stores true/false positive/negative counts for each NE type.
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        
        # Calculating accuracy slightly different as I think it is more useful 
        # Otherwise for rare tags, the true negative counts will completely skew results
        self.correct = 0
        self.target_total = 0

    def get_precision(self):
        return self.tp / float(self.tp + self.fp)

    def get_recall(self):
        return self.tp / float(self.tp + self.fn)

    def get_fscore(self):
        return 2*(self.get_precision()*self.get_recall())/(self.get_precision() + self.get_recall())
    
    def get_accuracy(self):
        return self.correct / self.target_total



class evaluator(object):
    
    def __init__(self, tag_names):

        # create a dictionary that holds a counter for each class
        self.tag_names = tag_names
        self.tag_counts = {}
        for c in self.tag_names:
            self.tag_counts[c] = neTypeCounts()
            
    def evalByTag(self, indices, tags, tagVoc):
        flat_idx, flat_tags = indices.view(-1), tags.view(-1)
        
        
        # switch this to
        # for pred, target in zip(flat_idx, flat_tags):
        for i in range(len(flat_idx)):
            pred, target = int(flat_idx[i]), int(flat_tags[i])
            

            # correct predictions
            if pred == target:
                
                self.tag_counts[tagVoc.itos[target]].tp += 1
                self.tag_counts[tagVoc.itos[target]].correct += 1
                
            # account for false positives and false negatives
            # if target = 2 and pred = 3
            # then we add a false negative to 2 and we add a false positive to 3 
            else:
                self.tag_counts[tagVoc.itos[target]].fn += 1
                self.tag_counts[tagVoc.itos[pred]].fp += 1
                
                    
            # add to total 
            self.tag_counts[tagVoc.itos[target]].target_total += 1
    
