import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch import autograd
import logging


from datareader import DataReader, START_TAG, END_TAG, PAD_IND, END_IND, START_IND, ROOT_IND
class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size,device):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transition.data.zero_()
        self.transition.data[START_IND,:] = torch.tensor(-10000)
        #self.transition.data[ROOT_IND,:] = torch.tensor(1000)
        self.transition.data[:,END_IND]  = torch.tensor(-10000)

    def forward(self, feats):
        """
        Forward propagation.
        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        #feats = feats[:,1:]
        self.batch_size = feats.size()[0]
        self.timesteps = feats.size()[1]

        emission_scores = self.emission(feats)  # (batch_size, timesteps, tagset_size)
        emission_scores = emission_scores.unsqueeze(3).expand(self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size)  # (batch_size, timesteps, tagset_size, tagset_size)
        #emission_scores[:,1:,:,:] = emission_scores[:,1:,:,:]+self.transition.unsqueeze(0).unsqueeze(0)
        #emission_scores[0,2] = emission_scores[0,2] +  self.transition
        crf_scores = torch.cat([emission_scores[:,0,:,:].unsqueeze(1),emission_scores[:,1:,:,:]+self.transition.unsqueeze(0).unsqueeze(0)],dim=1)
        #logging.info("Crf scores nasil gozukuyor bakalim")
        #logging.info(emission_scores[0,0])
        #logging.info(emission_scores[0,2])
        #logging.info("Transitions")
        #logging.info(self.transition)
        #logging.info(crf_scores[0,0])
        #logging.info(crf_scores[0,2])
        #logging.info(emission_scores[0,1]+self.transition)
        #crf_scores = emission_scores[:,1:,:,:] + self.transition.unsqueeze(0).unsqueeze(0)
            # (batch_size, timesteps, tagset_size, tagset_size)
        #mask[:,1:-1,:3,:] = 1
        #crf_scores = crf_scores.masked_fill(mask,-100)
        return crf_scores


class CRFLoss(nn.Module):
    
    def __init__(self,args,START_TAG = "[SOS]", END_TAG   = "[EOS]",device='cpu'):
        super(CRFLoss, self).__init__()
        self.device = device
        self.tagset_size = args['ner_cats']
        print("end index : {}".format(END_IND))
        print("start index : {}".format(START_IND))
        print("Tag set size : {}".format(self.tagset_size))
        self.START_TAG = START_TAG
        self.END_TAG = END_TAG
    
    def _log_sum_exp(self,tensor, dim):
        """
        Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.
        :param tensor: tensor
        :param dim: dimension to calculate log-sum-exp of
        :return: log-sum-exp
        """
        m, _ = torch.max(tensor, dim)
        m_expanded = m.unsqueeze(dim).expand_as(tensor)
        return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))

    ## gets crf scores shape (BATCH_SIZE, WORD_SIZES, TAG_SET, TAG_SET)
    def forward(self,scores, targets, lengths):
        """
            _log_sum_exp(all scores ) - gold_score

            lengths are used to ignore the loss at the paddings!!
            assumes that the sentences are sorted by length
        """

        ## this calculation assumes that the first target, i.e target[0]
        ## no it does no!
        #logging.info("Targets")
        #logging.info(targets[-1]//(scores.size()[2]))
        lengths = lengths   
        targets = targets[:,:]
        scores = scores[:,:]
        targets = targets.unsqueeze(2)
        batch_size = scores.size()[0]
        #scores_ =  scores.view(scores.size()[0],scores.size()[1],-1)
        score_before_sum = torch.gather(scores.reshape(scores.size()[0],scores.size()[1],-1), 2 , targets).squeeze(2)

        score_before_sums = pack_padded_sequence(score_before_sum,lengths, batch_first = True)
        #print(score_before_sum[0])
        gold_score = score_before_sums[0].sum()
        
        
        ## forward score : initialize from start tag
        forward_scores = torch.zeros(batch_size,self.tagset_size).to(self.device)
        forward_scores[:batch_size] = self._log_sum_exp(scores[:,0,:,:],dim=2)
        ## burada  hangisi  dogru emin   degilim index1-> index2 or  opposite?
        ## i think  opposite  is correct
        #forward_scores[:batch_size] = scores[:,0,:,self.tag2ind[self.START_TAG]]
        
        
        ## forward score unsqueeze 2ydi 1 yaptim cunku ilk index next tag olarak 
        ## kurguluyorum
        for i in range(1,scores.size()[1]):
            batch_size_t = sum([1 if lengths[x]>i else 0 for x in range(lengths.size()[0])])
            forward_scores[:batch_size_t] =\
                self._log_sum_exp(scores[:batch_size_t,i,:,:]\
                    +forward_scores[:batch_size_t].unsqueeze(1),dim=2)
        all_scores = forward_scores[:,END_IND].sum()
        loss = all_scores - gold_score
        #loss = loss/batch_size
        return loss

