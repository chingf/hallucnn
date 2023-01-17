import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from predify.utils.training import train_pcoders, eval_pcoders

from models.networks_2022 import BranchedNetwork
from data.NoisyDataset import NoisyDataset, FullNoisyDataset

# This is bad practice! But the warnings are real annoying
import warnings
warnings.filterwarnings("ignore")

exp = 'pnet'
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
activations_dir = f'{engram_dir}3_activations/{exp}/'
pickles_dir = f'{engram_dir}pickles/{exp}_denoising/'
os.makedirs(pickles_dir, exist_ok=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')


def pearsonr_sim(A, B):
    if torch.is_tensor(A):
        A = A.numpy()
    if torch.is_tensor(B):
        B = B.numpy()
    A = A.astype(float)
    B = B.astype(float)
    A = A.flatten()
    B = B.flatten()
    pear, _ = pearsonr(A, B)
    return pear

def eval_accuracy(results):
    """ t is the timestep to eval """
    accuracies = [] # for timesteps {0,1,2,3,4}
    n_timesteps = 4
    labels = np.array(results['label'])
    for t in range(n_timesteps+1):
        acc_t = np.mean((results[f'{t}_output'] == labels))
        accuracies.append(acc_t)
    return accuracies

def eval_correlations(results, dist_func, accs):
    labels = np.array(results['label'])
    idxs = np.arange(labels.size)
    
    popln_shuffle = []
    popln_sim = []
    popln_timestep = []
    popln_layer = []
    valid_score = []
    
    layers = ['conv1', 'conv2', 'conv3', 'conv4_W', 'conv5_W', 'fc6_W']
    
    n_timesteps = 4
    for t in range(n_timesteps+1):
        for l in layers:
            for i in idxs:
                noisy_activ = results[f'{l}_{t}_activations'][i]
                shuff_idx = np.random.choice(idxs)
                shuff_activ = results[f'{l}_{t}_clean_activations'][shuff_idx]
                clean_activ = results[f'{l}_{t}_clean_activations'][i]
                noisy_activ = noisy_activ.flatten()
                shuff_activ = shuff_activ.flatten()
                clean_activ = clean_activ.flatten()
                dist = dist_func(noisy_activ, clean_activ)
                shuff_dist = dist_func(noisy_activ, shuff_activ)
                popln_shuffle.append(shuff_dist)
                popln_sim.append(dist)
                popln_timestep.append(t)
                popln_layer.append(l)
                valid_score.append(accs[t])
        
    results = {
        'popln_shuffle': popln_shuffle,
        'popln_sim': popln_sim,
        'popln_timestep': popln_timestep,
        'popln_layer': popln_layer,
        'valid_score': valid_score
        }
    return results

file_prefix = 'pearsonr'
dist_func = pearsonr_sim

bgs = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snrs = [-9., -6., -3., 0., 3.]
args = []
for _bg in bgs:
    for _snr in snrs:
        args.append((_bg, _snr))
task_number = int(sys.argv[1])
bg, snr = args[task_number]

bg_snr_activations_dir = f'{activations_dir}{bg}_snr{int(snr)}/'
results = {
    'popln_shuffle': [], 'popln_timestep': [], 'popln_sim': [],
    'popln_layer': [], 'valid_score': []}
for hdf5_file in os.listdir(bg_snr_activations_dir):
    hdf5_path = f'{bg_snr_activations_dir}{hdf5_file}'
    hdf5_data = h5py.File(hdf5_path, 'r')
    accs = eval_accuracy(hdf5_data)
    _results = eval_correlations(hdf5_data, dist_func, accs)
    for key in results.keys():
        results[key].extend(_results[key])
with open(f'{pickles_dir}{file_prefix}_{bg}_snr{snr}.p', 'wb') as f:
    pickle.dump(results, f)
