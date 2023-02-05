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

        if self.cgram_shuffle == 1: # preserves freq info, not temporal info
            self.shuffle_idxs = np.zeros((self.n_data, 164), dtype=int)
            for i in range(self.n_data):
                indices = np.arange(164, dtype=int)
                np.random.shuffle(indices)
                self.shuffle_idxs[i] = indices
        elif self.cgram_shuffle == 2: # preserves temporal info, not freq info
            self.shuffle_idxs = np.zeros((self.n_data, 400), dtype=int)
            for i in range(self.n_data):
                indices = np.arange(400, dtype=int)
                np.random.shuffle(indices)
                self.shuffle_idxs[i] = indices
        
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
 
class NoisySoundsDataset(Dataset):
    """
    Noisy versions of the clean sounds dataset.
    HDF_FILE is typically: {engram_dir}hyperparameter_pooled_training_dataset_random_order_noNulls
    bg can be: 'babble_8spkr' or 'auditory_scene' or 'pink_noise'
    snr can be: 'neg9', 'neg6', 'neg3', '0', or '3'
    """

    def __init__(
        self, hdf_file, subset=None, train=True,
        label_key='label_indices', scaling=1000,
        bg=None, snr=None, random_order=True
        ):

        self.hdf_file = hdf_file
        self.label_key = label_key
        self.scaling = scaling
        self.train = train
        self.f = h5py.File(hdf_file, 'r')
        self.random_order = random_order

        # Subset by background noise or SNR as desired
        path_to_wav = np.array(self.f['path_to_wav']).astype('U')
        valid_index = []
        for index, wav in enumerate(path_to_wav):
            if bg != None:
                if not wav.startswith(bg): continue
            if snr != None:
                if (wav.split('_')[2] != snr): continue
            valid_index.append(index)
        self.valid_index = np.array(valid_index)

        # Determine size of final dataset
        self.n_data = self.valid_index.size
        if subset is not None:
            if train: 
                self.n_data = int(self.n_data*subset)
            else:
                self.n_data = int(self.n_data * (1-subset))
                self.start_ind = int(self.n_data * subset)

        # Random order
        if random_order:
            if subset is not None:
                import warnings
                warnings.warn("Random order dataset not tested for subsets!")
            self.new_indices = np.arange(self.n_data)
            np.random.shuffle(self.new_indices)

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        
        if not self.train:
            idx = idx + self.start_ind # Adds offset for test set 
            if self.random_order:
                idx = self.new_indices[idx]
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if self.random_order:
                idx = [self.new_indices[i] for i in idx]

        new_idx = self.valid_index[idx] 
        item = np.array(self.f['data'][new_idx]).reshape((-1, 164, 400))*self.scaling
        label = self.f[self.label_key][new_idx]
        return torch.tensor(item), torch.tensor(label).type(torch.LongTensor)

class GammaNoiseDataset(Dataset):
    """
    Gamma-distributed cochleagrams. Default args are measured from distribution
    of values from the CleanSoundsDataset. Specifically, they are generated via
    scipy.stats.gamma.rvs(3.825, -6.595, 37.200).
    HDF_FILE is typically: {engram_dir}gammaNoise_reconstruction_training_set
    """

    def __init__(self, hdf_file, subset=None, train=True, label_key='labels'):
        self.hdf_file = hdf_file
        self.train = train
        self.f = h5py.File(hdf_file, 'r')
        self.label_key = label_key
        self.n_data, _, _ =  np.shape(self.f['data'])
        
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
        label = self.f[self.label_key][idx]
        return torch.tensor(item), torch.tensor(label).type(torch.LongTensor)
