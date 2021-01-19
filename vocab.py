"""
    An abstract vocab class for supporting mapping and unmappings
"""


class Vocab:

    def __init__(self, w2ind, unk_ind=None):
        self.w2ind = w2ind
        self.ind2w = [x for x in w2ind.keys()]
        self.unk_ind = unk_ind
        self.unk = unk

    def map(self, units):
        return [self.w2ind.get(x, self.unk_ind) for x in units]

    def unmap(self, idx):
        return [self.ind2w[i] for i in idx]

    def __len__(self):
        return len(self.w2ind)
