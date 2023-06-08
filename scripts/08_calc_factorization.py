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

# Arguments
netname = str(sys.argv[1]) # pnet
shuffle = True
auc = True
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
activations_dir = f'{engram_dir}3_activations/{netname}/'
pickles_dir = f'{engram_dir}pickles/'
pca_activations_dir = f'{engram_dir}4_activations_pca/{netname}/'
bg_types = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snr_types = [-9.0, -6.0, -3.0, 0.0, 3.0]

# Helper functions
def get_data(conv_idx, t, bg, snr):
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
    
    # Repeat for clean
    if conv_idx > 3:
        clean_activ = np.array(results[f'conv{conv_idx}_W_{t}_clean_activations'])
    else:
        clean_activ = np.array(results[f'conv{conv_idx}_{t}_clean_activations'])
    clean_activ = clean_activ.reshape((n_data, -1))
    
    return activ, clean_activ, np.array(results['label'])

def get_projection(activ, pca):
    activ_centered = activ - pca.mean_[None,:]
    projected_activ = activ_centered @ (pca.components_).T
    return projected_activ

def get_explained_var(activ, pca, auc=True):
    """ ACTIV should be of shape (N, DIMS)"""
    
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    K = np.argwhere(cum_var>0.9)[0].item()
    cum_var_K = cum_var[K]
    activ_centered = activ - pca.mean_[None,:]
    sample_size = activ.shape[0]
    if sample_size == 0: sample_size = 1
    total_var = np.sum(np.square(activ_centered))/sample_size
    projected_activ = activ_centered @ pca.components_.T
    sample_size = projected_activ.shape[0]-1
    if sample_size == 0: sample_size = 1
    explained_var = np.sum(np.square(projected_activ), axis=0)/sample_size
    explained_var = explained_var/total_var
    if auc:
        var_curve = np.cumsum(explained_var)
        clean_explained = np.trapz(var_curve, dx=1/var_curve.size)
    else:
        clean_explained = np.sum(explained_var[:K+1])
    return clean_explained

if __name__ == "__main__":
    # Measure factorization for each noise/layer/timestep
    bgs = []
    snrs = []
    convs = []
    ts = []
    factorization = []
    for bg in bg_types:
        for snr in snr_types:
            for conv_idx in [1,2,3,4,5]:
                for t in [0,1,2,3,4]:
                    print(f'{bg}, {snr}, conv {conv_idx}, t {t}')
                    
                    # Load data and PCA model
                    activ, clean_activ, label = get_data(conv_idx, t, bg, snr)
                    print(activ.shape)

                    if not shuffle:
                        clean_pca_path = f'{pca_activations_dir}PCA_{bg}_{snr}_clean_conv{conv_idx}_t{t}.p'
                    else:
                        # Dummy clean dataset-- actually a shuffle
                        clean_pca_path = f'{pca_activations_dir}PCA_{bg}_{snr}_shufflehalf_conv{conv_idx}_t{t}.p'
                        _activ = np.zeros(activ.shape) # Dummy noisy dataset
                        _activ[1::2] = np.copy(clean_activ[1::2])
                        _activ[::2] = np.copy(activ[::2])
                        _clean_activ = np.zeros(clean_activ.shape) # Dummy clean
                        _clean_activ[1::2] = np.copy(activ[1::2])
                        _clean_activ[::2] = np.copy(clean_activ[::2])
                        activ = _activ
                        clean_activ = clean_activ

                    with open(clean_pca_path, 'rb') as f:
                        clean_pca = pickle.load(f)
                        
                    # Calculate factorization ratio for each sample (vectorize!!)
                    for i in range(activ.shape[0]):
                        noise_var = get_explained_var(
                            activ[i].reshape((1,-1)), clean_pca, auc=auc)
                        clean_var = get_explained_var(
                            clean_activ[i].reshape((1,-1)), clean_pca, auc=auc)
                        bgs.append(bg)
                        snrs.append(snr)
                        convs.append(conv_idx)
                        ts.append(t)
                        factorization.append(noise_var/clean_var)

                    del clean_pca
                    gc.collect()
            
    df = pd.DataFrame({
        'BG': bgs,
        'SNR': snrs,
        'Conv': convs,
        'T': ts,
        'Factorization': factorization
        })
    os.makedirs(pickles_dir, exist_ok=True)
    pfile = 'factorization.p'
    if auc:
        pfile = 'auc_' + pfile
    if shuffle:
        pfile = 'shuffle_' + pfile
    pfile = f'{pickles_dir}{netname}_{pfile}'
    with open(pfile, 'wb') as f:
        pickle.dump(df, f)

