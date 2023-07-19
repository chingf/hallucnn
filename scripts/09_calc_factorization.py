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

# Arguments
netname = str(sys.argv[1]) # pnet
sample_tag = str(sys.argv[2]) # n5000_sample0
if len(sys.argv) > 3:
    shuffle_seed = int(sys.argv[3])
    print(f'Shuffle-label PCA with seed {shuffle_seed}')
    shuffle = True
else:
    shuffle = False
auc = True
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
train_activations_dir = f'{engram_dir}3_train_activations/{netname}_{sample_tag}/'
pca_activations_dir = f'{engram_dir}4_train_prototype_PCA/{netname}_{sample_tag}/'
pickles_dir = f'{engram_dir}pickles/'
bg_types = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snr_types = [-9.0, -6.0, -3.0, 0.0, 3.0]
var_threshold = 0.9

# Helper functions
def get_cpu_usage():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t --giga').readlines()[-1].split()[1:])

    # Memory usage
    p_used = round((used_memory/total_memory) * 100, 2)
    print(f"RAM {used_memory} GB, {p_used}% used")

def get_train_data(conv_idx, t):
    activ = []
    label = []
    utterance = []
    for bg in bg_types:
        for snr in snr_types:
            print(f'{bg}, {snr}, conv {conv_idx}, t {t}')
            activ_dir = f'{train_activations_dir}{bg}_snr{int(snr)}/'
            results_files = [f for f in os.listdir(activ_dir) if 'pt' in f]
            for result_file in results_files:
                result_filepath = f'{activ_dir}{result_file}'
                with h5py.File(result_filepath, 'r') as results:
                    if conv_idx == 6:
                        _activ = np.array(results[f'fc6_W_{t}_activations'])
                    elif conv_idx > 3:
                        _activ = np.array(results[f'conv{conv_idx}_W_{t}_activations'])
                    else:
                        _activ = np.array(results[f'conv{conv_idx}_{t}_activations'])
                    _label = np.array(results['label'])
                    _clean_index = np.array(results['clean_index'])
                _activ = _activ.reshape((_activ.shape[0], -1))
            activ.append(_activ)
            label.append(_label)
            utterance.append(_clean_index)
    gc.collect()
    activ = np.vstack(activ)
    label = np.concatenate(label)
    utterance = np.concatenate(utterance).astype(int)
    return activ, label, utterance

def get_explained_var(centered_activ, pca, auc=True):
    """ ACTIV should be of shape (N, DIMS)"""
    
    sample_size = centered_activ.shape[0]
    projected_activ = centered_activ @ pca.components_.T
    total_var = np.sum(np.mean(np.square(projected_activ), axis=0))
    var_by_component = np.mean(np.square(projected_activ), axis=0)/total_var
    if auc:
        var_curve = np.cumsum(var_by_component)
        explained_var = np.trapz(var_curve, dx=1/var_curve.size)
    else:
        pca_cum_var = np.cumsum(pca.explained_variance_ratio_)
        K = np.argwhere(pca_cum_var>var_threshold)[0].item()
        explained_var = np.sum(var_by_component[:K+1])
    return explained_var 

def main():
    # Measure factorization for each noise/layer/timestep
    bgs = []
    snrs = []
    convs = []
    ts = []
    factorization = []

    for conv_idx in [1,2,3,4,5,6]:
        # Load PCA model and the utterance-prototype vectors from t = 0
        prototypes_fname = f'utterance_prototypes_conv{conv_idx}_t0'
        if shuffle:
            prototypes_fname += f'_shuffle{shuffle_seed}'
        prototypes_fname = f'{pca_activations_dir}{prototypes_fname}.p'
        with open(prototypes_fname, 'rb') as f:
            prototype_results = pickle.load(f)
        utterances_to_use = prototype_results['utterances']
        prototypes = prototype_results['prototypes']
        pca_fname = f'PCA_conv{conv_idx}_t0'
        if shuffle:
            pca_fname += f'_shuffle{shuffle_seed}' 
        with open(f'{pca_activations_dir}{pca_fname}.p', 'rb') as f:
            pca = pickle.load(f)

        # Iterate over timesteps of predictive processing
        for t in [0,1,2,3,4]:
            activ, _, utterance = get_train_data(conv_idx, t)
            if shuffle:
                np.random.seed(shuffle_seed+1)
                np.random.shuffle(utterance)

            # Calculate factorization for each utterance-centered datapoint
            var_ratios = []
            for u_idx, u in enumerate(utterances_to_use):
                activ_indices = utterance==u
                if np.sum(activ_indices) == 0:
                    continue
                prototype = prototypes[u_idx]
                centered_activ = activ[activ_indices] - prototype[None,:]
                l_var_ratio = get_explained_var(centered_activ, pca, auc=auc)
                if not auc:
                    l_var_ratio = l_var_ratio / var_threshold
                convs.append(conv_idx)
                ts.append(t)
                factorization.append(l_var_ratio)
        
            del activ
            del centered_activ
            gc.collect()
            get_cpu_usage()
            
    df = pd.DataFrame({
        'Conv': convs,
        'T': ts,
        'Factorization': factorization
        })
    os.makedirs(pickles_dir, exist_ok=True)
    pfile = f'factorization_{sample_tag}.p'
    if auc:
        pfile = 'auc_' + pfile
    if shuffle:
        pfile = 'shuffle_' + pfile
    pfile = f'{pickles_dir}{netname}_{pfile}'
    with open(pfile, 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()

