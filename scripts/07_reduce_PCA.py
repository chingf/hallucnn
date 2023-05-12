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
n_components = { # Chosen for 95% variance explained at t=0 for each conv_idx
    1: 2200, 2: 4200, 3: 4200, 4: 5000, 5: 5000}

##### HELPER FUNCTIONS #####
def get_and_reduce_data(conv_idx, t, pca):
    X = []
    bgs = []
    snrs = []
    dset_idxs = []
    n_total_data = 0
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
            activ = activ.reshape((n_data, -1))
            activ = pca.transform(activ)
            activ = activ[:, :n_components[conv_idx]]
            activ = list(activ)
            _dset_idxs = list(range(n_data))
            X.extend(activ)
            snrs.extend([snr]*n_data)
            bgs.extend([bg]*n_data)
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
   
    return X, bgs, snrs, dset_idxs

def reduce_and_save(conv_idx, t):
    pca_filename = f'FullPCAmodel_conv{conv_idx}_t{t}'
    with open(f'{pca_activations_dir}{pca_filename}.p', 'rb') as f:
        pca = pickle.load(f)
    X_pca, bgs, snrs, dset_idxs = get_and_reduce_data(conv_idx, t, pca)
    data_filename = f'data_conv{conv_idx}_t{t}'
    with h5py.File(f'{pca_activations_dir}{data_filename}.hdf5', 'x') as f_out:
        data_dict = {}
        data_dict['X_pca'] = f_out.create_dataset('X_pca', data=X_pca)
        bgs_ascii = [n.encode("ascii", "ignore") for n in bgs]
        data_dict['bgs'] = f_out.create_dataset('bgs', data=bgs_ascii, dtype='S10')
        data_dict['snrs'] = f_out.create_dataset('snrs', data=snrs)
        data_dict['dset_idxs'] = f_out.create_dataset('dset_idxs', data=dset_idxs)

##### MAIN CALL #####

if __name__ == "__main__":
    for conv_idx in [1, 2, 3, 4, 5]:
        for t in [0, 1, 2, 3, 4]:
            if conv_idx==1 and t==1: continue
            print(f'====== PROCESSING LAYER {conv_idx}, TIMESTEP {t} ======')
            reduce_and_save(conv_idx, t)
            gc.collect()

