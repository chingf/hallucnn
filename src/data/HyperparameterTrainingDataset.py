import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'

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
        bg=None, snr=None, random_order=False
        ):

        self.hdf_file = hdf_file
        self.label_key = label_key
        self.scaling = scaling
        self.train = train
        f = h5py.File(hdf_file, 'r')
        self.random_order = random_order

        # Janky conversion
        bg = convert_bg_string(bg)
        snr = convert_snr_string(snr)
        print(bg)
        print(snr)

        # Subset by background noise or SNR as desired
        path_to_wav = np.array(f['path_to_wav']).astype('U')
        valid_index = []
        for index, wav in enumerate(path_to_wav):
            if bg != None:
                if not wav.startswith(bg): continue
            if snr != None:
                if (wav.split('_')[2] != snr): continue
            valid_index.append(index)
        self.valid_index = np.array(valid_index)

        # Retain the correct indices to the corresponding clean dataset
        all_clean_indices = f'{engram_dir}indices_of_clean_corresponding_to_hyperparameter_pooled.npy'
        all_clean_indices = np.load(all_clean_indices)
        self.corresponding_clean_indices = all_clean_indices[self.valid_index]

        # Determine size of final dataset
        self.n_data = self.valid_index.size
        self.start_ind = 0
        if subset is not None:
            if train: 
                self.n_data = int(self.n_data*subset)
            else:
                n_data = int(self.n_data * (1-subset))
                self.start_ind = int(self.n_data * subset)
                self.n_data = n_data

        # Random order
        if random_order:
            self.new_indices = np.arange(self.n_data)
            np.random.shuffle(self.new_indices)

    def __len__(self):
        return self.n_data

    def open_hdf5(self):
        self.f = h5py.File(self.hdf_file, 'r')

    def __getitem__(self, idx):
        if not hasattr(self, 'f'): self.open_hdf5()
        # Indices may be shuffled into random order. Also add offset if test set.
        if torch.is_tensor(idx) and self.random_order:
            idx = [self.new_indices[i] + self.start_ind for i in idx]
        elif self.random_order:
            idx = self.new_indices[idx] + self.start_ind
        if torch.is_tensor(idx):
            idx = idx.tolist()

        new_idx = self.valid_index[idx] 
        item = np.array(self.f['data'][new_idx]).reshape((-1, 164, 400))*self.scaling
        label = self.f[self.label_key][new_idx]
        return torch.tensor(item), torch.tensor(label).type(torch.LongTensor)

class MergedNoisyDataset(Dataset):
    """
    More straightforward loading if you want all the noisy sounds together.
    """

    def __init__(self, subset=None, train=True):
        hdf_file = f'{engram_dir}'
        hdf_file += 'hyperparameter_pooled_training_dataset_random_order_noNulls.hdf5'
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

class TmpNoisyDatasetForReconstructionTraining(Dataset):
    """
    Contains 2967 samples per BG/SNR pair, to make a total of 44505 sounds.
    Assumed 90/10 train/test split.
    """

    def __init__(
        self, hdf_file, train=True, label_key='label_indices', scaling=1000,
        random_seed=0
        ):

        np.random.seed(random_seed)
        self.hdf_file = hdf_file
        self.train = train
        self.label_key = label_key
        self.scaling = scaling
        f = h5py.File(hdf_file, 'r')

        # Subset by background noise or SNR as desired
        bgs = ['auditory_scene', 'babble_8spkr', 'pink_noise']
        snrs = ['neg9', 'neg6', 'neg3', '0', '3']
        train_indices=[]; train_bgs=[]; train_snrs=[]
        test_indices=[]; test_bgs=[]; test_snrs=[]
        path_to_wav = np.array(f['path_to_wav']).astype('U')
        for bg in bgs:
            for snr in snrs:
                valid_index = []
                for index, wav in enumerate(path_to_wav):
                    if bg != None:
                        if not wav.startswith(bg): continue
                    if snr != None:
                        if (wav.split('_')[2] != snr): continue
                    valid_index.append(index)
                valid_index = np.array(valid_index)
                np.random.shuffle(valid_index)
                train_indices.extend(valid_index[:2670].tolist())
                test_indices.extend(valid_index[2670:2967].tolist())
                train_bgs.extend([bg]*2670); train_snrs.extend([snr]*2670)
                test_bgs.extend([bg]*297); test_snrs.extend([snr]*297)
        self.train_indices = np.array(train_indices)
        self.test_indices = np.array(test_indices)
        self.train_bgs = np.array(train_bgs)
        self.train_snrs = np.array(train_snrs)
        self.test_bgs = np.array(test_bgs)
        self.test_snrs = np.array(test_snrs)
        train_shuffle = np.arange(self.train_indices.size)
        test_shuffle = np.arange(self.test_indices.size)
        np.random.shuffle(train_shuffle)
        np.random.shuffle(test_shuffle)
        self.train_indices = self.train_indices[train_shuffle]
        self.test_indices = self.test_indices[test_shuffle]
        self.train_bgs = self.train_bgs[train_shuffle]
        self.train_snrs = self.train_snrs[train_shuffle]
        self.test_bgs = self.train_bgs[test_shuffle]
        self.test_snrs = self.train_snrs[test_shuffle]

        if train:
            self.valid_index = self.train_indices
        else:
            self.valid_index = self.test_indices
        self.n_data = self.valid_index.size

        np.random.seed()

    def __len__(self):
        return self.n_data

    def open_hdf5(self):
        self.f = h5py.File(self.hdf_file, 'r')

    def __getitem__(self, idx):
        if not hasattr(self, 'f'): self.open_hdf5()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        new_idx = self.valid_index[idx] 
        item = np.array(self.f['data'][new_idx]).reshape((-1, 164, 400))*self.scaling
        label = self.f[self.label_key][new_idx]
        return torch.tensor(item), torch.tensor(label).type(torch.LongTensor)

def convert_bg_string(bg):
    if bg == 'AudScene':
        bg = 'auditory_scene'
    elif bg == 'Babble8Spkr':
        bg = 'babble_8spkr'
    elif bg == 'pinkNoise':
        bg = 'pink_noise'
    return bg

def convert_snr_string(snr):
    if snr == -9.:
        snr = 'neg9'
    elif snr == -6.:
        snr = 'neg6'
    elif snr == -3.:
        snr = 'neg3'
    elif snr == 0.:
        snr = '0'
    elif snr == 3.:
        snr = '3'
    return snr


