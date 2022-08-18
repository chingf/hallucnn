import os
import torch
import pickle
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
#engram_dir = '/mnt/smb/locker/issa-locker/users/Erica/'

class MergedNoisyDataset(Dataset):

    def __init__(self, subset=None, train=True):
        hdf_file = f'{engram_dir}hyperparameter_pooled_training_dataset_random_order_noNulls.hdf5'
        self.f = h5py.File(hdf_file, 'r')
        self.n_data, _ = np.shape(self.f['data'])
        self.SCALING = 1000
        self.train = train
        if subset is not None:
            if train:
                self.n_data = int(self.n_data * subset)
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

        items = np.array(self.f['data'][idx]).reshape((-1, 164, 400))*self.SCALING
        labels = np.array(self.f['label_indices'][idx])
        return torch.tensor(items), torch.tensor(labels).type(torch.LongTensor)

