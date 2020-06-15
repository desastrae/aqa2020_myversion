#!/bin/python

# Create pytorch nn modules for further use

import torch
from torch import nn
from transformer.Encoder import Encoder


class SimpleNN(nn.Module):
    """ A class that inherits nn.Module"""
    def __init__(self, input_size=3, hidden_size=4, output_size=1):
        super(SimpleNN, self).__init__()

        """ Define all the layers here"""
        self.sigmoid = nn.Sigmoid()
        self.linear_a = nn.Linear(input_size, hidden_size)
        self.linear_b = nn.Linear(hidden_size, hidden_size)
        self.linear_c = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids):

        """Forward pass"""
        # input_ids size =  (batch x 3)
        output = self.linear_a(input_ids)  # tensor size: (batch x 4)
        output = self.sigmoid(output)  # non-linearity
        output = self.linear_b(output)  # tensor size: (batch x 4)
        output = self.sigmoid(output)  # non-linearity
        output = self.linear_c(output)  # tensor size (batch x 1)
        output = self.sigmoid(output)  # tensor size (batch x 1)

        return output


class TwoTextClassifier(nn.Module):
    def __init__(self, vocab_size, isize=512, hsize=512, output_size=1):
        super(TwoTextClassifier, self).__init__()
        self.encoder = Encoder(isize=isize, ahsize=hsize, num_layer=6, nwd=vocab_size)
        self.linear = nn.Linear(isize*2, 1)
        self.linear2 = nn.Linear(isize, 1)

    def forward(self, input1_ids, input2_ids, input1_mask, input2_mask,pred=0):
        if pred == 0:
            encoded_input1 = self.encoder(input1_ids, input1_mask)
            encoded_input2 = self.encoder(input2_ids, input2_mask)
            input1_rep = encoded_input1[:, 0, :]
            input2_rep = encoded_input2[:, 0, :]
            # takes input a tensor of size batch x 1 x (2*isize)
            pred1 = torch.sigmoid(self.linear(torch.cat((input1_rep, input2_rep), dim=1)))
            return pred1

        elif pred == 1:
            joint_encoded_input = self.encoder(torch.cat((input1_ids, input2_ids), dim=1),
                                               torch.cat((input1_mask, input2_mask), dim=2))
            joint_rep = joint_encoded_input[:, 0, :]
            pred2 = torch.sigmoid(self.linear2(joint_rep))
            return pred2


class WordEmbedding_with_scaling_factor_issue(nn.Module):

    def __init__(self, vocab_size, isize=512, hsize=512, window_size=5):
        super(WordEmbedding, self).__init__()

        self.WE_layer = nn.Embedding(vocab_size, hsize)
        self.WE_matrix = self.WE_layer.weight # nn.Parameter - can be used as torch tensor
        # torch.nn.init.uniform_(self.WE_layer.weight,0,1) # sample weights from a uniform distribution on [0,1]
       
        self.middle = int(window_size/2)
        self.scaling_factor = isize  

    def forward(self, input_ids, mode="skip"):

        main_word = input_ids[:, self.middle].unsqueeze(-1)  # batch_size x 1
        # batch_size x (window_size-1)
        context_words = torch.cat((input_ids[:, :self.middle], input_ids[:, self.middle+1:]), 1)

        ''' get embeddings of main word and context words '''
        main_emb = self.WE_layer(main_word)  # batch_size x 1 x hsize
        context_emb = self.WE_layer(context_words)  # batch_size x (window_size-1) x hsize

        if mode == "skip":  # SKIPGRAM
            ''' SKIP-gram : Given a word; predict the words in its context to the left and to the right '''
            # dot-product between main_word and context_words : batch_size x 1 x (window_size-1)
            ontext_scores = torch.exp(torch.matmul(main_emb, context_emb.transpose(1, 2)) / self.scaling_factor)
            # dot-product between main_word and all_words : batch_size x 1 x vocab_size
            all_skipgram_scores = torch.exp(main_emb.matmul(self.WE_matrix.transpose(0, 1)) / self.scaling_factor)

            # Note: scaling factor helps the dot product value to be limited;
            # we are exp(dot_product) further; for example, exp(80) is so large - becomes inf in python
  
            denominator_skipgram = all_skipgram_scores.sum(2).unsqueeze(-1)  # batch_size x 1 x 1
            context_log_probs = (context_scores / denominator_skipgram).squeeze(1)  # batch_size x (window_size-1)
            context_log_probs = torch.log(context_log_probs)

            # Note: you can divide a tensor of (axbxc) with (axbx1)[element-wise division];
            # It just divides all the values in dimension c
            return context_log_probs

        else: # CBOW
            ''' CBOW : Given context words; predict the main_word '''
            # sum the context word embeddings
            sum_of_context = context_emb.sum(1).unsqueeze(1)  # batch_size x 1 x hsize

            # Find dot product between main_word embedding and the (sum of embeddings of context words)
            # batch_size x 1 x 1
            main_word_score = torch.exp(main_emb.matmul(sum_of_context.transpose(1, 2))/self.scaling_factor)

            # Find dot product between (sum of embeddings of context words) and all possible word embeddings
            # affinity scores between avg_context_word and all_words : batch_size x 1 x vocab_size
            all_cbow_scores = torch.exp(main_emb.matmul(self.WE_matrix.transpose(0, 1)) / self.scaling_factor)

            denominator_cbow = all_cbow_scores.sum(2).unsqueeze(-1)
            # batch_size x 1
            cbow_log_prob = torch.log(main_word_score / denominator_cbow).squeeze(1)
            return cbow_log_prob


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, isize=512, hsize=512, window_size=5):
        super(WordEmbedding, self).__init__()

        self.WE_layer = nn.Embedding(vocab_size, hsize)
        self.linear_layer = nn.Linear(hsize, vocab_size)
        # sample weights from a uniform distribution on [0,1]
        torch.nn.init.uniform_(self.WE_layer.weight, 0, 1)
        # Share weights between prediction layer and word embedding layer
        self.linear_layer.weight.data = self.WE_layer.weight.data

        self.middle = int(window_size/2)
        self.scaling_factor = isize

    def forward(self, input_ids, mode="skip"):

        main_word = input_ids[:, self.middle].unsqueeze(-1)  # batch_size x 1
        # batch_size x (window_size-1)
        context_words = torch.cat((input_ids[:, :self.middle], input_ids[:, self.middle+1:]), 1)

        main_emb = self.WE_layer(main_word) # batch_size x 1 x hsize

        if mode == "skip":  # SKIPGRAM
            ''' SKIP get embeddings of main word; 
            use it to predict all probs [i.e, linear_layer+torch.softmax+torch.log]; 
            pick the log_probs of words that are in context '''
            all_scores = self.linear_layer(main_emb).squeeze(1)  # batch x 1 x vocab
            all_probs = torch.softmax(all_scores, dim=-1)
            all_log_probs = torch.log(all_probs).squeeze(1)  # batch x vocab

            # Will gather the values at the index tensor [context_words] :batch x (window_size-1)
            # ==> takes only log_probs(context_words)
            context_log_probs = all_log_probs.gather(1, context_words)
            return context_log_probs

        else:  # CBOW
            ''' CBOW get embeddings of context words, sum them up ; 
            use it to predict the all probs[ i.e, linear_layer+torch.softmax + torch.log]; 
            pick the log_probs of the main word;'''
            # sum the context word embeddings
            sum_of_context = context_emb.sum(1).unsqueeze(1)  # batch_size x 1 x hsize

            all_cbow_scores = self.linear_layer(sum_of_context)  # batch_size x 1 x vocab_size
            all_cbow_probs = torch.softmax(all_cbow_scores, dim=-1)
            all_cbow_log_probs = torch.log(all_cbow_probs)

            ''' NOTE: you can combine 2-3 functions at once; here I did one after another, 
            so you would understand that functios are applied one after another'''
            # Will gather the values at the index tensor [main_word] :batch x (1) ==> takes only log_probs(main_word)
            cbow_log_prob = all_cbow_log_probs.gather(1, main_word)

            return cbow_log_prob


class SquadQA(nn.Module):
    def __init__(self, vocab_size, isize=512, hsize=512):
        super(SquadQA, self).__init__()

        self.encoder = Encoder(isize=isize, ahsize=hsize, num_layer=6, nwd=vocab_size)
        self.linear = nn.Linear(isize, isize)
        self.linear_endpos = nn.Linear(isize, isize)
        self.linear2 = nn.Linear(2*isize, isize)  # used to obtain start-position informed cls rep

        '''TODO'''

    def forward(self, input_ids, input_mask, start_positions=None):
        output = self.encoder(input_ids, input_mask)  # batch_size x seq_length x hsize
        cls_rep = output[:, 0:1, :]  # batch_size x 1 x hsize

        projected_output = self.linear(output) # batch_size x seq_length x hsize

        # get dot product between #cls vector and projected output vectors
        start_position_scores = cls_rep.matmul(projected_output.transpose(1, 2))
        # batch_size x 1 x seq_length
        start_position_probs = torch.softmax(start_position_scores, dim=-1)

        '''
        # Given start_position, question and paragrah, predict end_position:
        # argmax(function) is a non-continous function, this can't be back-propagated; can only be used during evaluation
        if start_positions is None: 
           val, start_positions = start_positions_prob.max(-1)
           start_positions=start_positions.squeeze(1)
           start_positions.requires_grad=False
 
        # Get 1-hot start position
        startpos_1hot = torch.zeros(input_ids.size(0),input_ids.size(1))
        startpos_1hot = startpos_1hot.scatter(1,start_positions,1).unsqueeze(1) # batch_sizex1xseq_length

        startpos_rep = startpos_1hot.matmul(projected_output) # pick the start pos rep
        informed_cls_rep = torch.cat((cls_rep,startpos_rep),dim=-1) #batch_size x 1 x (2xhsize)
        informed_cls_rep = self.linear2(informed_cls_rep)
        
        # get dot product between #cls vector and projected output vectors
        end_position_scores = informed_cls_rep.matmul(output.transpose(1,2)) 
        end_position_probs = torch.softmax(end_position_scores,dim=-1) # batch_size x 1 x seq_length

        '''
        # Given  just question, paragraph, predict end_position

        # batch_size x seq_length x hsize
        projected_endpos_output = self.linear_endpos(output)
        # get dot product between #cls vector and projected output vectors
        end_position_scores = cls_rep.matmul(projected_endpos_output.transpose(1, 2))
        # batch_size x 1 x seq_length
        end_position_probs = torch.softmax(start_position_scores, dim=-1)

        return torch.cat((start_position_probs, end_position_probs), dim=1)
        

'''
import logging
logger = logging.getLogger(__name__)
'''
