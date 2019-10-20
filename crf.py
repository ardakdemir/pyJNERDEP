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

class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))
        self.transition.data.zero_()

    def forward(self, feats):
        """
        Forward propagation.
        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.size()[0]
        self.timesteps = feats.size()[1]

        emission_scores = self.emission(feats)  # (batch_size, timesteps, tagset_size)
        emission_scores = emission_scores.unsqueeze(2).expand(self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size)  # (batch_size, timesteps, tagset_size, tagset_size)

        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(
            0)  # (batch_size, timesteps, tagset_size, tagset_size)
        return crf_scores


class CRFLoss(nn.Module):
    def __init__(self,tag_set,START_TAG = "[SOS]", END_TAG   = "[EOS]"):
        super(CRFLoss, self).__init__()
        self.tag2ind = tag_set
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
        ## no it does not!
        targets = targets.unsqueeze(2)
        batch_size = scores.size()[0]
        print(scores[0].shape)
        #scores_ =  scores.view(scores.size()[0],scores.size()[1],-1)
        score_before_sum = torch.gather(scores.view(scores.size()[0],scores.size()[1],-1), 2 , targets).squeeze(2)

        score_before_sums = pack_padded_sequence(score_before_sum,lengths, batch_first = True)
        #print(score_before_sum[0])
        gold_score = score_before_sums[0].sum()
        ## forward score : initialize from start tag
        forward_scores = torch.zeros(batch_size, len(self.tag2ind))
        forward_scores[:batch_size] = scores[:,0,self.tag2ind[self.START_TAG],:]
        for i in range(1, scores.size()[1]):
            batch_size_t = sum([1 if lengths[x]>i else 0 for x in range(lengths.size()[0])])
            forward_scores[:batch_size_t] =\
                self._log_sum_exp(scores[:batch_size_t,i,:,:]\
                    +forward_scores[:batch_size_t].unsqueeze(2),dim=1)
        all_scores = forward_scores[:,self.tag2ind[self.END_TAG]].sum()
        loss = all_scores - gold_score
        loss = loss/batch_size
        return loss
