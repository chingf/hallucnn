import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

class CleanSoundsDataset(Dataset):
    """ Clean sounds dataset. """

    def __init__(self, hdf_file, subset=None):
        self.hdf_file = hdf_file
        f = h5py.File(hdf_file, 'r')
        self.data = torch.tensor(f['data']).reshape((-1, 164, 400))
        self.labels = torch.tensor(f['labels'])
        if subset is not None:
            self.data = self.data[:subset]
            self.labels = self.labels[:subset]
        self.n_data, self.height, self.width = self.data.shape

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]

