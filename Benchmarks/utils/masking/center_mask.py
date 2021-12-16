import numpy as np


def createCenterMaskPercent(slice, percent):
    mask = np.ones(slice.shape)
    i = 0
    currentPercent = 1

    while currentPercent > percent:
        i += 1
        mask[0:i, :, :] = 0
        mask[slice.shape[0] - i:, :] = 0
        mask[:, 0:i] = 0
        mask[:, slice.shape[1] - i:] = 0

        currentPercent = np.count_nonzero(mask) / mask.size

    return mask
