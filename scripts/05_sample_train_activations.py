#!/usr/bin/env python
# coding: utf-8

# In[2]:

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
from data.HyperparameterTrainingDataset import NoisySoundsDataset

##### ARGS ######
pnet_name = str(sys.argv[1]) # pnet
subsample_seed = int(sys.argv[2])
n_units_to_sample = int(sys.argv[3])

# Set up directory paths
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
activations_dir = f'{engram_dir}3_train_activations/{pnet_name}/'
new_activations_dir = f'{engram_dir}3_train_activations/{pnet_name}' \
    f'_n{n_units_to_sample}_sample{subsample_seed}/'
pca_activations_dir = f'{engram_dir}4_train_prototype_PCA/{pnet_name}/'

# Initialize useful parameters
bg_types = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snr_types = [-9.0, -6.0, -3.0, 0.0, 3.0]
n_units_per_layer = {
    1: (96, 55, 134), 2: (256, 14, 34),
    3: (512, 7, 17), 4: (1024, 7, 17),
    5: (512, 7, 17), 6: (4096,)
    }
n_labels = 531 # Label 0 is for negative samples; should not be used
np.random.seed(subsample_seed)
units_to_sample = {}
for layer in n_units_per_layer.keys():
    n_units = np.prod(n_units_per_layer[layer])
    if n_units > n_units_to_sample:
        units_to_sample[layer] = np.random.choice(
            n_units, size=n_units_to_sample, replace=False)
    else:
        units_to_sample[layer] = np.arange(n_units)
np.random.seed()

# Helper function
def get_layer_idx(key):
    if key.startswith('conv1'): return 1
    elif key.startswith('conv2'): return 2
    elif key.startswith('conv3'): return 3
    elif key.startswith('conv4'): return 4
    elif key.startswith('conv5'): return 5
    elif key.startswith('fc6'): return 6
    else:
        raise ValueError('Invalid layer key')

# Iterate through noise types and load activations
for bg in bg_types:
    for snr in snr_types:
        activ_dir = f'{activations_dir}{bg}_snr{int(snr)}/'
        new_activ_dir = f'{new_activations_dir}{bg}_snr{int(snr)}/'
        os.makedirs(new_activ_dir, exist_ok=True)
        for results_file in os.listdir(activ_dir):
            if 'pt' not in results_file: continue
            results_filepath = f'{activ_dir}{results_file}'
            new_results_filepath = f'{new_activ_dir}{results_file}'
            print(f'Processing {results_filepath}')
            with h5py.File(results_filepath, 'r') as f_src:
                with h5py.File(new_results_filepath, 'x') as f_out:
                    for key in f_src.keys():
                        if 'activations' not in key:
                            f_src.copy(f_src[key], f_out, key)
                            continue
                        data = np.array(f_src[key])
                        l = get_layer_idx(key)
                        print(f'{key} with {data.shape}-- should be {n_units_per_layer[l]}')
                        print(data.shape[1:])
                        assert(data.shape[1:] == n_units_per_layer[l])
                        data = data.reshape((data.shape[0], -1))
                        data = data[:, units_to_sample[l]]
                        f_out.create_dataset(key, data=data)

