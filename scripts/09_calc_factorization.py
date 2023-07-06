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
shuffle = False
auc = False
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
train_activations_dir = f'{engram_dir}3_train_activations/{netname}/'
validation_activations_dir = f'{engram_dir}3_validation_activations/{netname}/'
pca_activations_dir = f'{engram_dir}4_train_prototype_PCA/{netname}/'
pickles_dir = f'{engram_dir}pickles/'
bg_types = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snr_types = [-9.0, -6.0, -3.0, 0.0, 3.0]

# Helper functions
def get_cpu_usage():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t --giga').readlines()[-1].split()[1:])

    # Memory usage
    p_used = round((used_memory/total_memory) * 100, 2)
    print(f"RAM {used_memory} GB, {p_used}% used")

def get_valid_data(conv_idx, t, bg, snr):
    activ_dir = f'{validation_activations_dir}{bg}_snr{int(snr)}/'
    for results_file in os.listdir(activ_dir):
        results_filepath = f'{activ_dir}{results_file}'
        results = h5py.File(results_filepath, 'r')
    if conv_idx > 3:
        activ = np.array(results[f'conv{conv_idx}_W_{t}_activations'])
    else:
        activ = np.array(results[f'conv{conv_idx}_{t}_activations'])
    n_data = activ.shape[0]
    activ = activ.reshape((n_data, -1))
    return activ, np.array(results['label'])

def get_projection(activ, pca):
    activ_centered = activ - pca.mean_[None,:]
    projected_activ = activ_centered @ (pca.components_).T
    return projected_activ

def get_explained_var(centered_activ, pca, auc=True):
    """ ACTIV should be of shape (N, DIMS)"""
    
    sample_size = centered_activ.shape[0]
    total_var = np.sum(np.square(centered_activ))/sample_size
    projected_activ = centered_activ @ pca.components_.T
    var_by_component = np.sum(np.square(projected_activ), axis=0)/sample_size
    var_by_component = var_by_component/total_var
    if auc:
        var_curve = np.cumsum(var_by_component)
        explained_var = np.trapz(var_curve, dx=1/var_curve.size)
    else:
        pca_cum_var = np.cumsum(pca.explained_variance_ratio_)
        K = np.argwhere(pca_cum_var>0.9)[0].item()
        explained_var = np.sum(var_by_component[:K+1])
    return explained_var 

def main():
    # Measure factorization for each noise/layer/timestep
    bgs = []
    snrs = []
    convs = []
    ts = []
    factorization = []

    for conv_idx in [1,2,3,4,5]:

        # Load PCA model and the prototype vectors from t = 0
        prototypes_fname = f'prototypes_conv{conv_idx}_t0'
        prototypes_fname = f'{pca_activations_dir}{prototypes_fname}.p'
        with open(prototypes_fname, 'rb') as f:
            prototype_results = pickle.load(f)
        labels_to_use = prototype_results['labels']
        prototypes = prototype_results['prototypes']
        pca_filename = f'PCA_conv{conv_idx}_t0'
        with open(f'{pca_activations_dir}{pca_filename}.p', 'rb') as f:
            pca = pickle.load(f)

        # Iterate over timesteps of predictive processing
        for t in [0,1,2,3,4]:
            activ = []
            label = []
            for bg in bg_types:
                for snr in snr_types:
                    print(f'{bg}, {snr}, conv {conv_idx}, t {t}')
                    _activ, _label = get_valid_data(conv_idx, t, bg, snr)
                    activ.append(_activ)
                    label.append(_label)
            del _activ
            del _label
            gc.collect()
            activ = np.vstack(activ)
            label = np.concatenate(label)

            # Calculate factorization for each centered label
            var_ratios = []
            for l_idx, l in enumerate(labels_to_use):
                activ_indices = label==l
                if np.sum(activ_indices) == 0:
                    continue
                prototype = prototypes[l_idx]
                centered_activ = activ[activ_indices] - prototype[None,:]
                l_var_ratio = get_explained_var(centered_activ, pca, auc=auc)
                if not auc:
                    l_var_ratio = l_var_ratio / 0.9
                convs.append(conv_idx)
                ts.append(t)
                factorization.append(l_var_ratio)
        
            del activ
            del label
            gc.collect()
            
    df = pd.DataFrame({
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

if __name__ == "__main__":
    main()
