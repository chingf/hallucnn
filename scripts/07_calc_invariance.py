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
    clean_noise_dist = []
    utterance_dist = []
    clean_clean_dist = []
    for bg in bg_types:
        print(bg)
        for snr in snr_types:
            print(snr)
            for conv_idx in [1,2,3,4,5]:
                for t in [0,1,2,3,4]:
                    print(f'Layer {conv_idx}, time {t}')
                    activ, clean_activ, label = get_data(conv_idx, t, bg, snr)
                    bgs.append(bg)
                    snrs.append(snr)
                    convs.append(conv_idx)
                    ts.append(t)
                    
                    # Clean-noisy distance for same audio clip
                    _clean_noise_dist = []
                    for idx in range(len(activ)):
                        _clean_noise_dist.append(
                            np.square(activ[idx] - clean_activ[idx]))
                    _clean_noise_dist = np.mean(_clean_noise_dist, axis=0)
                    _clean_noise_dist = np.sum(_clean_noise_dist)
                    clean_noise_dist.append(_clean_noise_dist)
                    
                    # Distance between utterances of the same word
                    _utterance_dist = []
                    for l in np.unique(label):
                        utterance_activ = clean_activ[label==l]
                        for i in range(utterance_activ.shape[0]):
                            for j in range(0,i):
                                _utterance_dist.append(np.square(
                                    utterance_activ[i] - utterance_activ[j]))
                    _utterance_dist = np.mean(_utterance_dist, axis=0)
                    _utterance_dist = np.sum(_utterance_dist)
                    utterance_dist.append(_utterance_dist)
                    
                    # Distance between any two clean audio clips
                    _clean_clean_dist = []
                    for idx in range(len(activ)):
                        rand_idx = np.random.choice(len(activ))
                        _clean_clean_dist.append(np.square(
                            clean_activ[idx] - clean_activ[rand_idx]))
                    _clean_clean_dist = np.mean(_clean_clean_dist, axis=0)
                    _clean_clean_dist = np.sum(_clean_clean_dist)
                    clean_clean_dist.append(_clean_clean_dist)
    
    invariance_df = pd.DataFrame({
        'BG': bgs,
        'SNR': snrs,
        'Conv': convs,
        'T': ts,
        'Dist': clean_noise_dist,
        'Invariance by Utterance': np.array(clean_noise_dist)/np.array(utterance_dist),
        'Invariance by Radius': np.array(clean_noise_dist)/np.array(clean_clean_dist),
        })

    os.makedirs(pickles_dir, exist_ok=True)
    pfile = f'{pickles_dir}{netname}_invariance.p'
    with open(pfile, 'wb') as f:
        pickle.dump(invariance_df, f)
    
