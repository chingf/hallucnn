import os
import torch
import pickle
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'

class LargeNoisyDataset(Dataset):
    ''' WSJ Training Utterances '''

    def __init__(self, bg, snr, orig_dset='WSJ'):
        hdf_file = f'{engram_dir}{bg}_{int(snr)}_fixed.hdf5'
        label_key = f'{engram_dir}PsychophysicsWord2017W_999c6fc475be1e82e114ab9865aa5459e4fd329d.__META_key.npy'
        self.label_key = np.load(label_key)
        self.f = h5py.File(hdf_file, 'r')
        self.n_data = 5000 #, __ =  np.shape(self.f['data'])
        self.SCALING = 1000

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        items = np.array(self.f['data'][idx]).reshape((-1, 164, 400))*self.SCALING
        labels = np.array(self.f['label_indices'][idx])
        return torch.tensor(items), torch.tensor(labels).type(torch.LongTensor)

class NoisyDataset(Dataset):
    '''
    WSJ Validation Utterances. Wrapper around FullNoisyDataset to allow for
    selection of subsets. Options include:
    bgs = ['pinkNoise' 'AudScene', 'Babble8Spkr']
    snrs = [-9.0, -6.0, -3.0, 0.0, 3.0]
    '''

    def __init__(self, bg, snr, orig_dset='WSJ'):
        fullnoisydata = FullNoisyDataset()
        idxs = fullnoisydata.orig_dset == orig_dset

        if bg != 'pinkNoise':
            idxs = np.logical_and(idxs, fullnoisydata.bg == bg)
            idxs = np.logical_and(idxs, fullnoisydata.snr == snr)
            self.noisy_in = fullnoisydata.noisy_in[idxs]
            self.bg = fullnoisydata.bg[idxs]
            self.snr = fullnoisydata.snr[idxs]
            self.net_mistakes = fullnoisydata.net_mistakes[idxs]
        else:
            with h5py.File(
                f'{engram_dir}psychophysics_{bg}_{int(snr)}_fixed.hdf5', 'r'
                ) as f:
                self.noisy_in = np.array(f['data']).reshape((-1, 164, 400))*1000
                _idxs = np.zeros(idxs.shape).astype(bool)
                _idxs[:self.noisy_in.shape[0]] = True
                idxs = np.logical_and(idxs, _idxs)
                self.noisy_in = self.noisy_in[idxs[:self.noisy_in.shape[0]]]
                n_data = self.noisy_in.shape[0]
                self.bg = np.array([bg] * n_data)
                self.snr = np.array([snr] * n_data)
                self.net_mistakes = np.array([None] * n_data)
                assert('Timit' not in fullnoisydata.orig_dset[idxs])
            
        self.clean_in = fullnoisydata.clean_in[idxs]
        self.labels = fullnoisydata.labels[idxs]
        self.orig_dset = fullnoisydata.orig_dset[idxs]
        self.n_data = self.labels.size

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.noisy_in[idx]
        label = self.labels[idx]
        return torch.tensor(item), torch.tensor(label)

class FullNoisyDataset():
    ''' WSJ Validation Utterances '''

    def __init__(self):
        self._load_psychophysics_file()
        self._load_net_mistakes()
        self._load_labels()
        self._load_bg()
        self._load_snr()
        self._load_orig_dset()
        self._load_clean_in()

    def _load_psychophysics_file(self):
        f_in = h5py.File(
            f"{engram_dir}PsychophysicsWord2017W_not_resampled.hdf5",
            'r')
        self.f_metadata = np.load(
            f"{engram_dir}PsychophysicsWord2017W_999c6fc475be1e82e114ab9865aa5459e4fd329d.__META.npy",
            'r')
        self.f_key = np.load(
            f"{engram_dir}PsychophysicsWord2017W_999c6fc475be1e82e114ab9865aa5459e4fd329d.__META_key.npy",
            'r')
        self.noisy_in = np.array(f_in['data']).reshape((-1, 164, 400))

    def _load_net_mistakes(self):
        with open(f"{engram_dir}PsychophysicsWord2017W_net_performance.p", 'rb') as f:
            self.net_mistakes = pickle.load(f)['net_mistakes']

    def _load_labels(self):
        labels = []
        for word in self.f_metadata['word']:
            idx = np.argwhere(self.f_key == word)
            if len(idx) == 0:
                labels.append(-1)
            else:
                labels.append(idx.item())
        self.labels = np.array(labels) + 1

    def _load_bg(self):
        bg = []
        for _bg in self.f_metadata['bg']:
            bg.append(str(_bg, 'utf-8'))
        self.bg = np.array(bg)

    def _load_snr(self):
        snr = []
        for _snr in self.f_metadata['snr']:
            _snr = str(_snr, 'utf-8')
            if 'inf' in _snr:
                _snr = np.inf
            elif 'neg' in _snr:
                if '3' in _snr:
                    _snr = -3
                elif '6' in _snr:
                    _snr = -6
                elif '9' in _snr:
                    _snr = -9
                else:
                    raise ValueError('Not found')
            else:
                if '0' in _snr:
                    _snr = 0
                elif '3' in _snr:
                    _snr = 3
                else:
                    raise ValueError('Not found')
            snr.append(_snr)
        self.snr = np.array(snr)

    def _load_orig_dset(self):
        orig_dset = []
        for _orig_dset in self.f_metadata['orig_dset']:
            _orig_dset = str(_orig_dset, 'utf-8')
            _orig_dset = 'WSJ' if 'WSJ' in _orig_dset else 'Timit'
            orig_dset.append(_orig_dset)
        self.orig_dset = np.array(orig_dset)

    def _load_clean_in(self):
        """ Loads Psychophysics clean 2017 """

        cochleagrams_clean = []
        cochleagrams = []
        for batch_ii in range(0,15300,100):
            cgram_dir = f'{engram_dir}cgrams_for_noise_robustness_analysis/PsychophysicsWord2017W_clean/'
            hdf5_path = f'{cgram_dir}batch_'+str(batch_ii)+'_to_'+str(batch_ii+100)+'.hdf5'
            with h5py.File(hdf5_path, 'r') as f_in:
                cochleagrams += list(f_in['data'])
        self.clean_in = np.array(cochleagrams)
        n_data = self.labels.size
        self.clean_in = self.clean_in[:n_data]


