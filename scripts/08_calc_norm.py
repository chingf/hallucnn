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
sample_tag = str(sys.argv[2]) # n5000_sample0
if len(sys.argv) > 3:
    shuffle_seed = int(sys.argv[3])
    print(f'Shuffle-label invariance with seed {shuffle_seed}')
    shuffle = True
else:
    shuffle = False
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
train_activations_dir = f'{engram_dir}3_train_activations/{netname}_{sample_tag}/'
validation_activations_dir = f'{engram_dir}3_validation_activations/{netname}/'
pickles_dir = f'{engram_dir}pickles/'
bg_types = ['pinkNoise', 'AudScene', 'Babble8Spkr']
bg_types = ['AudScene']
snr_types = [-9.0, -6.0, -3.0, 0.0, 3.0]
snr_types = [-9.0, -6.0, -3.0]

# Helper functions
def get_cpu_usage():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t --giga').readlines()[-1].split()[1:])

    # Memory usage
    p_used = round((used_memory/total_memory) * 100, 2)
    print(f"RAM {used_memory} GB, {p_used}% used")

def get_train_data(conv_idx, t, shuffle=False):
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
            results_files = [f for f in os.listdir(activ_dir) if f'pt' in f]
            if len(results_files) == 0: continue
            for result_file in results_files:
                result_filepath = f'{activ_dir}{result_file}'
                with h5py.File(result_filepath, 'r') as results:
                    if conv_idx == 6:
                        activ = np.array(results[f'fc6_W_{t}_activations'])
                    elif conv_idx > 3:
                        activ = np.array(results[f'conv{conv_idx}_W_{t}_activations'])
                    else:
                        activ = np.array(results[f'conv{conv_idx}_{t}_activations'])
                    clean_index = np.array(results['clean_index'])
                    if shuffle:
                        np.random.seed(shuffle_seed)
                        np.random.shuffle(clean_index)
                    for _dset_idx, _clean_index in enumerate(clean_index):
                        _clean_index = int(_clean_index)
                        _activ = activ[_dset_idx].flatten()
                        activity[_clean_index].append(_activ)
        break

    # Remove zero-rows
    new_activity = []
    for i in range(len(activity)):
        if len(activity[i]) < 2: continue
        new_activity.append(activity[i])
    activity = new_activity

    #if prune and conv_idx==1:
    #    max_samples = 2000
    #    if len(activity) > max_samples:
    #        indices_to_use = np.random.choice(len(activity), size=max_samples)
    #        new_activity = []
    #        for i in indices_to_use:
    #            new_activity.append(activity[i])
    #        activity = new_activity

    # Reset seed if needed
    if shuffle:
        np.random.seed()
    return activity

def main():
    # Measure factorization for each noise/layer/timestep
    convs = []
    ts = []
    invariance = []

    for conv_idx in [1,2,3,4,5,6]:
        # Iterate over timesteps of predictive processing
        for t in [0,1,2,3,4]:
            activity = get_train_data(conv_idx, t, shuffle=shuffle)
            flattened_activity = [row for a in activity for row in a]
            flattened_activity = np.array(flattened_activity)

            # Collect data
            activity_norm = np.linalg.norm(flattened_activity, axis=1).tolist()
            convs.extend([conv_idx]*len(activity_norm))
            ts.extend([t]*len(activity_norm))
            invariance.extend(activity_norm)
            del flattened_activity
            gc.collect()
            
    df = pd.DataFrame({
        'Conv': convs,
        'T': ts,
        'Invariance': invariance
        })
    os.makedirs(pickles_dir, exist_ok=True)
    pfile = f'norm_{sample_tag}.p'
    if shuffle:
        pfile = 'shuffle_' + pfile
    pfile = f'{pickles_dir}{netname}_{pfile}'
    with open(pfile, 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()

