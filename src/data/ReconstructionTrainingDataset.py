import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

class CleanSoundsDataset(Dataset):
    """
    Clean sounds dataset from WSJ, but excludes the psychophysics.
    HDF_FILE is typically: {engram_dir}clean_reconstruction_training_set
    """

    def __init__(self, hdf_file, subset=None, train=True, label_key='label_indices', scaling=1000):
        self.hdf_file = hdf_file
        self.label_key = label_key
        self.scaling = scaling
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
        item = np.array(self.f['data'][idx]).reshape((-1, 164, 400))*self.scaling
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
        bg=None, snr=None
        ):

        self.hdf_file = hdf_file
        self.label_key = label_key
        self.scaling = scaling
        self.train = train
        self.f = h5py.File(hdf_file, 'r')

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

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        
        if not self.train:
            idx = idx + self.start_ind # Adds offset for test set 
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        new_idx = self.valid_index[idx] 
        item = np.array(self.f['data'][new_idx]).reshape((-1, 164, 400))*self.scaling
        label = self.f[self.label_key][new_idx]
        return torch.tensor(item), torch.tensor(label).type(torch.LongTensor)
  
