from utils.BenchmarkDataset import BenchmarkDataset
from torch.utils.data import Dataset, DataLoader
from Undersampling import undersamplingKSP
from utils.masking.center_mask import createCenterMaskPercent
from utils.inv_fourier import ifft2c
from utils.coil_combine import rss
# from Undersampling import undersamplingKSP
from Undersampling.undersamplingKSP import *
#from utils.masking.center_mask import createCenterMaskPercent
import h5py
from utils.to_tensor import to_tensor
import pandas as pd

#print(t2_data.head())

DATA_ROOT_DIR = '/data/project/fastMRI/Brain/multicoil_train/'
t2_data = pd.read_csv('/project/mpattadk/KSProject/T2_multicoil_train.csv')

train_csv_path = '/project/mpattadk/KSProject/T2_multicoil_train.csv'

def get_dataloader(
        dataset: Dataset,
        path_to_csv: str,
        batch_size: int = 1,
        num_workers: int = 4,
):
    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)

    dataset = dataset(df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader

dataloader = get_dataloader(dataset=BenchmarkDataset, path_to_csv=train_csv_path)
print(len(dataloader))

for data,label,slices in dataloader:
    for i in range (slices):
        data_slice = data[:,i,:,:,:]
        masked_data = createCenterMaskPercent(data_slice,0.90)
        print(masked_data)
        under_sampled_data = undersamplingKSP.performUndersamplingKSP(data_slice,masked_data)
        inv_fourier_image = ifft2c(under_sampled_data)
        inv_fourier_image = to_tensor(inv_fourier_image)
        coil_combined_rss = rss(inv_fourier_image)
        break
    break






# print(t2_data.head(5))
# h5File = h5py.File(t2_data['file'][0], "r")
# print(list(h5File.keys()))
# kspace = h5File['kspace'][()]
# ismrmrd_header = h5File['ismrmrd_header'][()]
# reconstruction_rss = h5File['reconstruction_rss'][()]
# print(ismrmrd_header)
# print('kspace shape: ', kspace.shape)
# coil1 = kspace[0:1, :, :, :]
# print('reconstruction_rss shape:', reconstruction_rss.shape)
#
# print('coil shape: ', coil1.shape)
# coil1 = coil1.reshape(16, 768, 396)
# print('coil shape: ', coil1.shape)
#
# #Masking
# masked = center_mask.createCenterMaskPercent(coil1, 0.8)
# print('masked shape: ', masked.shape)
# print(masked)
#
# #Undersampling
# #output_ = undersamplingKSP.performUndersamplingKSP(coil1, masked)
# #print('undersampling shape:', output_.shape)
# #print(output_)
#
#
# #output_tensor = to_tensor.to_tensor(output_)
#
# #coil_combine = coil_combine.rss(output_tensor, 1)



