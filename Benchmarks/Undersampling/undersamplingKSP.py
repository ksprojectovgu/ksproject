import numpy as np


def performUndersamplingKSP(fullKSPVol, mask):
    underKSPVol = np.multiply(fullKSPVol, mask)
    return underKSPVol
