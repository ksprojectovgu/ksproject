import pandas as pd
from Undersampling import undersamplingKSP
from utils.masking import center_mask
from utils import inv_fourier
from utils import coil_combine
from utils import to_tensor

import h5py

DATA_ROOT_DIR = '/data/project/fastMRI/Brain/multicoil_train/'

t2_data = pd.read_csv('/project/mpattadk/KSProject/T2_multicoil_train.csv')
print(t2_data.head())

h5File = h5py.File("/data/project/fastMRI/Brain/multicoil_train/file_brain_AXT2_200_2000394.h5", "r")
print(list(h5File.keys()))
kspace = h5File['kspace'][()]
ismrmrd_header = h5File['ismrmrd_header'][()]
reconstruction_rss = h5File['reconstruction_rss'][()]
print(ismrmrd_header)
print('kspace shape: ', kspace.shape)
coil1 = kspace[0:1, :, :, :]
print('reconstruction_rss shape:', reconstruction_rss.shape)

print('coil shape: ', coil1.shape)
coil1 = coil1.reshape(16, 768, 396)
print('coil shape: ', coil1.shape)

#Masking
masked = center_mask.createCenterMaskPercent(coil1, 0.8)
print('masked shape: ', masked.shape)
# print(mask)

#Undersampling
output_ = undersamplingKSP.performUndersamplingKSP(coil1, masked)
print('undersampling shape:', output_.shape)
print(output_)


output_tensor = to_tensor.to_tensor(output_)

coil_combine = coil_combine.rss(output_tensor, 1)



