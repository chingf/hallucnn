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
def get_data(conv_idx, t, sample_size=None):
    X = []
    bgs = []
    snrs = []
    dset_idxs = []
    n_total_data = 0
    n_sampled_data = 0
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
            n_total_data += n_data
            if sample_size != None:
                sample_idxs = np.random.choice(n_data, size=sample_size)
                activ = activ[sample_idxs]
                n_sampled_data += sample_size
                _dset_idxs = list(sample_idxs)
            else:
                n_sampled_data += n_data
                _dset_idxs = list(range(n_data))
            new_n_data = activ.shape[0]
            activ = list(activ.reshape((new_n_data, -1)))
            X.extend(activ)
            snrs.extend([snr]*new_n_data)
            bgs.extend([bg]*new_n_data)
            dset_idxs.extend(_dset_idxs)
            
            del results
            del activ
            gc.collect()

    idxs = np.arange(len(X))
    np.random.shuffle(idxs)

    X = np.array(X)[idxs]
    bgs = np.array(bgs)[idxs]
    snrs = np.array(snrs)[idxs]
    dset_idxs = np.array(dset_idxs)[idxs]
    
    print(f'Sampled {n_sampled_data}/{n_total_data} data')
    print(f'with {X.shape[1]} features')
    
    return X, bgs, snrs, dset_idxs

def get_cpu_usage():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t --giga').readlines()[-1].split()[1:])

    # Memory usage
    p_used = round((used_memory/total_memory) * 100, 2)
    print(f"RAM {used_memory} GB, {p_used}% used")

def fit_and_save(conv_idx, t):
    X, bgs, snrs, dset_idxs = get_data(conv_idx, t)
    print('Fitting Model...')
    pca = PCA()
    pca.fit(X)
    print(f'{pca.n_components_} components used for % variance explained:')
    print(np.sum(pca.explained_variance_ratio_))
    get_cpu_usage()


    os.makedirs(pca_activations_dir, exist_ok=True)
    pca_filename = f'FullPCAmodel_conv{conv_idx}_t{t}'
    with open(f'{pca_activations_dir}{pca_filename}.p', 'wb') as f:
        pickle.dump(pca, f, protocol=4)

##### MAIN CALL #####

if __name__ == "__main__":
    for conv_idx in [1, 2, 3, 4, 5]:
        for t in [0, 1, 2, 3, 4]:
            if conv_idx==1 and t==1: continue
            print(f'====== PROCESSING LAYER {conv_idx}, TIMESTEP {t} ======')
            fit_and_save(conv_idx, t)
            gc.collect()

