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
netname = str(sys.argv[1]) # pnet
if len(sys.argv) > 2 and str(sys.argv[2]) == 'shufflehalf':
    print('Shuffle-half PCA. Clean even + noisy odd for model fit.')
    shufflehalf = True
else:
    shufflehalf = False
    
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
                clean_activ = np.array(results[f'conv{conv_idx}_W_{t}_clean_activations'])
            else:
                clean_activ = np.array(results[f'conv{conv_idx}_{t}_clean_activations'])
            print(f'{bg}, snr {snr}, conv {conv_idx}, t {t}')
            print(clean_activ.shape)
            n_data = clean_activ.shape[0]
            clean_activ = clean_activ.reshape((n_data, -1))
            print(f'Runing PCA on clean with data shape {clean_activ.shape}')
            pca = PCA()
            pca.fit(clean_activ)
            pca_filename = f'PCA_{bg}_{snr}_clean_conv{conv_idx}_t{t}'
            with open(f'{pca_activations_dir}{pca_filename}.p', 'wb') as f:
                pickle.dump(pca, f, protocol=4)

def get_shufflehalf_PCA(conv_idx, t, pca_activations_dir):
    for bg in bg_types:
        for snr in snr_types:
            activ_dir = f'{activations_dir}{bg}_snr{int(snr)}/'
            for results_file in os.listdir(activ_dir):
                results_filepath = f'{activ_dir}{results_file}'
                results = h5py.File(results_filepath, 'r')
            if conv_idx > 3:
                activ = np.array(results[f'conv{conv_idx}_W_{t}_activations'])
                clean_activ = np.array(results[f'conv{conv_idx}_W_{t}_clean_activations'])
            else:
                activ = np.array(results[f'conv{conv_idx}_{t}_activations'])
                clean_activ = np.array(results[f'conv{conv_idx}_{t}_clean_activations'])
            n_data = activ.shape[0]
            activ = activ.reshape((n_data, -1))
            clean_activ = clean_activ.reshape((n_data, -1))
            shuffle_activ = np.zeros(activ.shape)
            shuffle_activ[::2] = np.copy(clean_activ[::2])
            shuffle_activ[1::2] = np.copy(activ[1::2])
            print(f'Runing shuffle PCA on {bg} {snr} with data shape {shuffle_activ.shape}')
            pca = PCA()
            pca.fit(shuffle_activ)
            pca_filename = f'PCA_{bg}_{snr}_shufflehalf_conv{conv_idx}_t{t}'
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
            if shufflehalf:
                get_shufflehalf_PCA(conv_idx, t, pca_activations_dir)
            else:
                get_data_and_fit_PCA(conv_idx, t, pca_activations_dir)
            gc.collect()

