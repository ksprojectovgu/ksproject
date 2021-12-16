import numpy as np

def ifft2c(x, shape=None, axes=(0,1), shiftAxes = (0,1), normalize=None): # originally was axes=(-2,-1), shiftAxes = None
    f = np.empty(x.shape, dtype=np.complex128)
    if(len(x.shape) == 4):
        for i in range(x.shape[-1]):
            for j in range(x.shape[-2]):
                f[:,:,j,i] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x[:,:,j,i]), s=shape, norm=normalize))
    elif(len(x.shape) == 3):
        for i in range(x.shape[-1]):
            f[:,:,i] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x[:,:,i]), s=shape, norm=normalize))
    else:
        f = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), s=shape, norm=normalize))
    return f