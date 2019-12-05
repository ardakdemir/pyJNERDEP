"""
    An abstract vocab class for supporting mapping and unmappings
"""

PAD = "[PAD]"
PAD_IND = 0

VOCAB_PREF = {PAD : PAD_IND}
class Vocab:

    def __init__(self,w2ind):
        self.w2ind = VOCAB_PREF + w2ind
        self.ind2w = [x for x in w2ind.keys()]

    def map(self,units):
        return [self.w2ind[x] for x in units]

    def unmap(self,idx):
        return [self.ind2w[i] for i in idx]
