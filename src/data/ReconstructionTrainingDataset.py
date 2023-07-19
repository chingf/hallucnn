import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

class CleanSoundsDataset(Dataset):
    """
    Clean sounds dataset from WSJ, but excludes the psychophysics.
    HDF_FILE is typically: {engram_dir}clean_reconstruction_training_set

    cgram_shuffle: 0: None, 1: frequency shuffle, 2: temporal shuffle
    """

    def __init__(
        self, hdf_file, subset=None, train=True, label_key='label_indices',
        scaling=1000, cgram_shuffle=0
        ):
        self.hdf_file = hdf_file
        self.label_key = label_key
        self.scaling = scaling
        self.train = train
        self.cgram_shuffle = cgram_shuffle
        self.f = h5py.File(hdf_file, 'r')
        self.n_data, __ =  np.shape(self.f['data'])

        if self.cgram_shuffle == 1: # preserves temporal info, not freq info
            self.shuffle_idxs = np.zeros((self.n_data, 164), dtype=int)
            for i in range(self.n_data):
                indices = np.arange(164, dtype=int)
                np.random.shuffle(indices)
                self.shuffle_idxs[i] = indices
        elif self.cgram_shuffle == 2: # preserves freq info, not temporal info
            self.shuffle_idxs = np.zeros((self.n_data, 400), dtype=int)
            for i in r ange(self.n_data):
                indices = np.arange(400, dtype=int)
                np.random.shuffle(indices)
                self.shuffle_idxs[i] = indices
       
        self.start_ind= 0
        if subset is not None:
            if train: 
                self.n_data = int(self.n_data*subset)
            else:
                self.n_data = int(self.n_data * (1-subset))
                self.start_ind = int(self.n_data * subset)

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        
        if not self.train:
            idx = idx + self.start_ind # Adds offset for test set 
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = np.array(self.f['data'][idx]).reshape((-1, 164, 400))*self.scaling
        if self.cgram_shuffle == 1:
            shuffle_idxs = self.shuffle_idxs[idx]
            if item.shape[0] == 1:
                item = item[:, shuffle_idxs]
            else:
                for batch in range(shuffle_idxs.shape[0]):
                    item[batch] = item[batch, shuffle_idxs[batch]]
        elif self.cgram_shuffle == 2:
            shuffle_idxs = self.shuffle_idxs[idx]
            if item.shape[0] == 1:
                item = item[:, :, shuffle_idxs]
            else:
                for batch in range(shuffle_idxs.shape[0]):
                    item[batch] = item[batch, :, shuffle_idxs[batch]]
        label = self.f[self.label_key][idx]
        return torch.tensor(item), torch.tensor(label).type(torch.LongTensor)
 
