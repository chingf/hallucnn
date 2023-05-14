import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import seaborn as sns
import datetime
from scipy.stats import sem
import matplotlib.cm as cm
import pathlib
import traceback
import gc
from data.ValidationDataset import NoisyDataset

##### ARGS ######
netname = 'pnet'
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
activations_dir = f'{engram_dir}3_activations/{netname}/'
pca_activations_dir = f'{engram_dir}4_activations_pca/{netname}/'
bg_types = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snr_types = [-9.0, -6.0, -3.0, 0.0, 3.0]

##### HELPER FUNCTIONS #####
def get_data_and_fit_PCA(conv_idx, t, pca_activations_dir):
    for bg in bg_types:
        for snr in snr_types:
            activ_dir = f'{activations_dir}{bg}_snr{int(snr)}/'
            for results_file in os.listdir(activ_dir):
                results_filepath = f'{activ_dir}{results_file}'
                results = h5py.File(results_filepath, 'r')
            break
            if conv_idx > 3:
                activ = np.array(results[f'conv{conv_idx}_W_{t}_activations'])
            else:
                activ = np.array(results[f'conv{conv_idx}_{t}_activations'])
            n_data = activ.shape[0]
            activ = activ.reshape((n_data, -1))
            print(f'Runing PCA on {bg} {snr} with data shape {activ.shape}')
            pca = PCA()
            pca.fit(activ)
            pca_filename = f'PCA_{bg}_{snr}_conv{conv_idx}_t{t}'
            with open(f'{pca_activations_dir}{pca_filename}.p', 'wb') as f:
                pickle.dump(pca, f, protocol=4)
    # Repeat for clean
    if conv_idx > 3:
        activ = np.array(results[f'conv{conv_idx}_W_{t}_clean_activations'])
    else:
        activ = np.array(results[f'conv{conv_idx}_{t}_clean_activations'])
    n_data = activ.shape[0]
    activ = activ.reshape((n_data, -1))
    print(f'Runing PCA on clean with data shape {activ.shape}')
    pca = PCA()
    pca.fit(activ)
    pca_filename = f'PCA_clean_conv{conv_idx}_t{t}'
    with open(f'{pca_activations_dir}{pca_filename}.p', 'wb') as f:
        pickle.dump(pca, f, protocol=4)   

def get_cpu_usage():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t --giga').readlines()[-1].split()[1:])

    # Memory usage
    p_used = round((used_memory/total_memory) * 100, 2)
    print(f"RAM {used_memory} GB, {p_used}% used")

##### MAIN CALL #####

if __name__ == "__main__":
    os.makedirs(pca_activations_dir, exist_ok=True)
    for conv_idx in [1, 2, 3, 4, 5]:
        for t in [0, 1, 2, 3, 4]:
            print(f'====== PROCESSING LAYER {conv_idx}, TIMESTEP {t} ======')
            get_data_and_fit_PCA(conv_idx, t, pca_activations_dir)
            gc.collect()

