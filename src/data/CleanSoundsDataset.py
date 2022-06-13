import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

class CleanSoundsDataset(Dataset):
    """ Clean sounds dataset. SCALED BY 1000 """

    def __init__(self, hdf_file, subset=None, scaling=1000):
        self.hdf_file = hdf_file
        self.scaling = scaling
        f = h5py.File(hdf_file, 'r')
        self.data = torch.tensor(
            np.array(f['data']).reshape((-1, 1, 164, 400))
            )
        self.labels = torch.tensor(f['labels'])
        if subset is not None:
            self.data = self.data[:subset]
            self.labels = self.labels[:subset]
        self.data = self.data * self.scaling
        self.n_data, self.n_channels, self.height, self.width = self.data.shape

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]

