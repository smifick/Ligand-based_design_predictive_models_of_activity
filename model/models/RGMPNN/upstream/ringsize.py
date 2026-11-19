import numpy as np


def ringsize(atom, rings):
    bits = np.zeros(7)
    id = atom.GetIdx()
    for ring in rings:
        a = len(ring)
        if a > 8:
            a = 9
        if id in ring:
            bits[a - 3] += 1
    return bits
