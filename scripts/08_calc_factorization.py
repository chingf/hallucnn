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

if __name__ == "__main__":
    bgs = []
    snrs = []
    convs = []
    ts = []
    factorization = []
    for bg in bg_types:
        for snr in snr_types:
            for conv_idx in [1,2,3,4,5]:
                for t in [0,1,2,3,4]:
                    activ, clean_activ, label = get_data(conv_idx, t, bg, snr)
                    bgs.append(bg)
                    snrs.append(snr)
                    convs.append(conv_idx)
                    ts.append(t)
                    
                    clean_pca_path = f'{pca_activations_dir}PCA_clean_conv{conv_idx}_t{t}.p'
                    with open(clean_pca_path, 'rb') as f:
                        clean_pca = pickle.load(f)
                    clean_cum_var = np.cumsum(clean_pca.explained_variance_ratio_)
                    K = np.argwhere(clean_cum_var>0.9)[0].item()
                    cum_var_K = clean_cum_var[K]
                    activ_centered = activ - clean_pca.mean_[None,:]
                    total_var = np.sum(np.square(activ_centered))/(activ.shape[0]-1)
                    explained_variance = []
                    for k in range(K+1):
                        var = np.sum(np.square(
                            activ_centered @ clean_pca.components_[k]
                            ))/(activ.shape[0]-1)
                        explained_variance.append(var)
                    clean_explained = np.sum(explained_variance)/total_var
                    _factorization = 1 - clean_explained/cum_var_K
                    factorization.append(_factorization)

    factorization_df = pd.DataFrame({
        'BG': bgs,
        'SNR': snrs,
        'Conv': convs,
        'T': ts,
        'Factorization': factorization,
        })

    os.makedirs(pickles_dir, exist_ok=True)
    pfile = f'{pickles_dir}{netname}_factorization.p'
    with open(pfile, 'wb') as f:
        pickle.dump(factorization_df, f)

