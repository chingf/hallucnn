import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import seaborn as sns
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator
from scipy.stats import pearsonr

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from predify.utils.training import train_pcoders, eval_pcoders

from models.networks_2022 import BranchedNetwork
from data.NoisyDataset import NoisyDataset, FullNoisyDataset

engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
pickle_dir = f'{engram_dir}pickles/'

bgs = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snrs = [-9.0, -6.0, -3.0, 0.0, 3.0]
args = []
for _bg in bgs:
    for _snr in snrs:
        args.append((_bg, _snr))

task_number = int(sys.argv[1])
bg, snr = args[task_number]
results = {'bg': [], 'snr': [], 'xc': [], 't': []}
dset = NoisyDataset(bg, snr)
n_data = len(dset)
for i in range(n_data):
    coch = dset[i][0].numpy()
    F, T = coch.shape
    for t in range(2, T//3):
        _xc = []
        for f in range(F):
            cc = np.corrcoef(coch[f,:-t], coch[f,t:])
            _xc.append(cc[0,1])
        results['bg'].append(bg)
        results['snr'].append(snr)
        results['xc'].append(np.mean(_xc))
        results['t'].append(t)
with open(f'{pickle_dir}{bg}_{snr}_stationarity.p', 'wb') as f:
    pickle.dump(results, f)

