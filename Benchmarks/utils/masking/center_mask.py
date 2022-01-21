import numpy as np
import math
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

def createCenterMaskPercent(slice, percent):
    mask = np.ones(slice[0].shape)
    i = 0
    currentPercent = 1
    while currentPercent > percent:
        i += 1
        mask[:,:,0:i] = 0
        mask[:,:,slice.shape[3]-i:] = 0

        mask[:, 0:i] = 0
        mask[:, slice.shape[2]-i:] = 0

        currentPercent = np.count_nonzero(mask)/mask.size

    return mask.reshape(1,slice.shape[1],slice.shape[2],slice.shape[3])



def createVardenMask1D(slice, percent, maxAmplitude4PDF, ROdir, returnPDF=False):
    mask = np.zeros(slice.shape)
    if ROdir == 2:
        percent = percent/2
        if slice.shape[0] == slice.shape[1]:
            mask, distfunc, randseed = _mask1DForROdir(mask, percent, maxAmplitude4PDF, 0)
            mask, _, _ = _mask1DForROdir(mask, percent, maxAmplitude4PDF, 1, distfunc, randseed)
        elif slice.shape[0] > slice.shape[1]:
            mask, distfunc, randseed = _mask1DForROdir(mask, percent, maxAmplitude4PDF, 0)
            dim_difference = slice.shape[0] - slice.shape[1]
            _distfunc = distfunc[dim_difference//2:(slice.shape[1]+dim_difference//2)]
            _randseed = randseed[dim_difference//2:(slice.shape[1]+dim_difference//2)]
            mask, _, _ = _mask1DForROdir(mask, percent, maxAmplitude4PDF, 1, _distfunc, _randseed)
        else:
            mask, distfunc, randseed = _mask1DForROdir(mask, percent, maxAmplitude4PDF, 1)
            dim_difference = slice.shape[1] - slice.shape[0]
            _distfunc = distfunc[dim_difference//2:(slice.shape[0]+dim_difference//2)]
            _randseed = randseed[dim_difference//2:(slice.shape[0]+dim_difference//2)]
            mask, _, _ = _mask1DForROdir(mask, percent, maxAmplitude4PDF, 0, _distfunc, _randseed)
    else:
        mask, distfunc, _ = _mask1DForROdir(mask, percent, maxAmplitude4PDF, ROdir)

    if returnPDF:
        if slice.shape[0] > slice.shape[1]:
            return mask, np.tile(distfunc,(slice.shape[1],1))
        else:
            return mask, np.tile(distfunc,(slice.shape[0],1))
    else:
        return mask

def _mask1DForROdir(mask, percent, maxAmplitude4PDF, ROdir, distfunc=None, randseed=None):
    shape = mask.shape[ROdir]
    if distfunc is None or randseed is None:
        #Random Numbers Seed
        randseed = np.random.random(shape)

        #Initialize variables
        x = np.array(range(shape))
        mu = np.floor(x.size/2)
        sigma = np.floor(x.size/2)

        currentPercent=1
        while currentPercent > percent:
            #Distribution function
            distfunc = maxAmplitude4PDF*math.sqrt(2*math.pi)*sigma*norm.pdf(x,mu,sigma);       # 0.8

            #Selection of k-space lines
            B = np.zeros(shape)
            #B[randseed>distfunc] = 0
            B[randseed<distfunc] = 1
            B[round(shape/2-shape/round(shape/4)):round(shape/2+shape/round(shape/4))] = 1
            currentPercent = np.count_nonzero(B)/B.size
            sigma = np.floor(0.95*sigma)
    else:
        #Selection of k-space lines
        B = np.zeros(shape)
        #B[randseed>distfunc] = 0
        B[randseed<distfunc] = 1
        B[round(shape/2-shape/round(shape/4)):round(shape/2+shape/round(shape/4))] = 1

    if ROdir == 0:
        mask[B==1,:] = 1
    else: #ROdir == 1:
        mask[:,B==1] = 1

    return mask, distfunc, randseed