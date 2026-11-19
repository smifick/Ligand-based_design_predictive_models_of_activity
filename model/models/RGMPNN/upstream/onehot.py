import numpy as np


def onehot(c, cat, other=False):
    length = len(cat)
    cat = dict(zip(cat,list(range(length))))
    if other == False:
        onehot = np.zeros(length)
        if c in cat:
            onehot[cat[c]] = 1
        assert onehot.sum() == 1
    if other == True:
        onehot = np.zeros(length+1)
        if c in cat:
            onehot[cat[c]] = 1
        else:
            onehot[-1] = 1
        assert onehot.sum() == 1
    return onehot
