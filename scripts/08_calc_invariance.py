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
if len(sys.argv) > 2:
    shuffle_seed = int(sys.argv[2])
    print(f'Shuffle-label invariance with seed {shuffle_seed}')
    shuffle = True
else:
    shuffle = False
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
train_activations_dir = f'{engram_dir}3_train_activations/{netname}/'
validation_activations_dir = f'{engram_dir}3_validation_activations/{netname}/'
pickles_dir = f'{engram_dir}pickles/'
bg_types = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snr_types = [-9.0, -6.0, -3.0, 0.0, 3.0]

# Helper functions
n_units_per_layer = {
    1: (96, 55, 134), 2: (256, 14, 34),
    3: (512, 7, 17), 4: (1024, 7, 17),
    5: (512, 7, 17)
    }
n_labels = 531 # Label 0 is for negative samples; should not be used

def get_cpu_usage():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t --giga').readlines()[-1].split()[1:])

    # Memory usage
    p_used = round((used_memory/total_memory) * 100, 2)
    print(f"RAM {used_memory} GB, {p_used}% used")

def get_data(conv_idx, t, shuffle=False):
    """
    Will return a (N, K, D) activity dataset.
        N: number of utterances
        K: augmentations of each utterance (noise)
        D: number of units, flattened
    """

    N = 5000 # number of unique utterances in the hyperparameter dataset
    activity = [[] for _ in range(N)]
    for bg in bg_types:
        for snr in snr_types:
            activ_dir = f'{train_activations_dir}{bg}_snr{int(snr)}/'
            for results_file in os.listdir(activ_dir):
                if 'pt' not in results_file: continue
                results_filepath = f'{activ_dir}{results_file}'
                with h5py.File(results_filepath, 'r') as results:
                    if conv_idx > 3:
                        activ = np.array(results[f'conv{conv_idx}_W_{t}_activations'])
                    else:
                        activ = np.array(results[f'conv{conv_idx}_{t}_activations'])
                    clean_index = np.array(results['clean_index'])
                    if shuffle:
                        np.random.seed(shuffle_seed)
                        np.random.shuffle(clean_index)
                    for _dset_idx, _clean_index in enumerate(clean_index):
                        activity[_clean_index].append(activ[_dset_idx].flatten())
    if shuffle:
        np.random.seed()
    return activity

def main():
    # Measure factorization for each noise/layer/timestep
    convs = []
    ts = []
    invariance = []

    for conv_idx in [1,2,3,4,5]:
        # Iterate over timesteps of predictive processing
        for t in [0,1,2,3,4]:
            activity = get_data(conv_idx, t, shuffle=shuffle)
            N, K, D = activity.shape
            var_noise = np.var(np.mean(activity, axis=1), axis=0).sum()
            var_all = np.var(activity.reshape((-1, D)), axis=0).sum()
            convs.append(conv_idx)
            ts.append(t)
            invariance.append(var_noise/var_all)
            del activity
            gc.collect()
            
    df = pd.DataFrame({
        'Conv': convs,
        'T': ts,
        'Invariance': invariance
        })
    os.makedirs(pickles_dir, exist_ok=True)
    pfile = 'invariance.p'
    if shuffle:
        pfile = 'shuffle_' + pfile
    pfile = f'{pickles_dir}{netname}_{pfile}'
    with open(pfile, 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()

