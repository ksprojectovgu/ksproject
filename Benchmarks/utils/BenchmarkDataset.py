from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
import os


class BenchmarkDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.root_path = '/data/project/fastMRI/Brain/multicoil_train/'

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        file_name = self.df.loc[idx, 'file']
        img_path = os.path.join(self.root_path, file_name)
        img,label = self.load_img(img_path)
        slices = img.shape[0]
        #print(slices)
        # num_slices = str(slices).split(',')
        # print(num_slices[0])
        #norm_img = self.normalize(img)


        return img,label,slices

    def load_img(self, file_path):
        h5File = h5py.File(file_path, "r")
        data = h5File['kspace'][()]
        label = h5File['reconstruction_rss'][()]
        return data, label

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
