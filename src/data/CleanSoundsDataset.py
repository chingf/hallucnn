import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

class CleanSoundsDataset(Dataset):
    """
    Clean sounds dataset from WSJ, but excludes the psychophysics.
    """

    def __init__(self, hdf_file, subset = None, train = True):
        self.hdf_file = hdf_file
        self.train = train
        self.f = h5py.File(hdf_file, 'r')
        self.n_data, __ =  np.shape(self.f['data'])
        
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
        item = np.array(self.f['data'][idx]).reshape((-1, 164, 400))
        label = self.f['labels'][idx]
        return torch.tensor(item), torch.tensor(label).type(torch.LongTensor)
  
class TrainCleanSoundsDataset(Dataset):
    """
    Clean sounds dataset from WSJ, but excludes the psychophysics.
    SCALED BY 1000
    """

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

class PsychophysicsCleanSoundsDataset(Dataset):
    """ Corresponding clean sounds to Psychophysics dataset """

    def __init__(self, clean_in, labels, orig_dset, exclude_timit=True, subset=None):
        if exclude_timit:
            new_clean_in = []
            for idx, orig_dset in enumerate(orig_dset):
                if orig_dset == 'WSJ':
                    new_clean_in.append(clean_in[idx])
            new_clean_in = np.array(new_clean_in)
            labels = labels[orig_dset=='WSJ']
            clean_in = new_clean_in
        n_samples = clean_in.shape[0]
        shuffle_idxs = np.arange(n_samples)
        np.random.shuffle(shuffle_idxs)
        clean_in = clean_in[shuffle_idxs]
        labels = labels.squeeze()
        self.labels = torch.tensor(labels[shuffle_idxs])
        self.data = torch.tensor(
            clean_in.reshape((n_samples, 1, 164, 400))
            )
        if subset is not None:
            self.data = self.data[:subset]
            self.labels = self.labels[:subset]
        self.n_data, self.n_channels, self.height, self.width = self.data.shape

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]
