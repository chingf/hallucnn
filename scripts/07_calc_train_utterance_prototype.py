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

##### ARGS ######
netname = str(sys.argv[1]) # pnet
if len(sys.argv) > 2:
    shuffle_seed = int(sys.argv[2])
    print(f'Shuffle-label PCA with seed {shuffle_seed}')
    shuffle = True
else:
    shuffle = False
    
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
activations_dir = f'{engram_dir}3_train_activations/{netname}/'
pca_activations_dir = f'{engram_dir}4_train_prototype_PCA/{netname}/'
bg_types = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snr_types = [-9.0, -6.0, -3.0, 0.0, 3.0]

##### HELPER FUNCTIONS #####
n_units_per_layer = {
    1: (96, 55, 134), 2: (256, 14, 34),
    3: (512, 7, 17), 4: (1024, 7, 17),
    5: (512, 7, 17)
    }
n_labels = 531 # Label 0 is for negative samples; should not be used
n_utterances = 5000

def get_data_and_fit_PCA(conv_idx, t, pca_activations_dir, shuffle=False):
    n_units = np.prod(n_units_per_layer[conv_idx])
    utterance_prototypes = [np.zeros(n_units) for l in range(n_utterances)]
    utterance_count = [0 for l in range(n_utterances)]
    for bg in bg_types:
        for snr in snr_types:
            activ_dir = f'{activations_dir}{bg}_snr{int(snr)}/'
            for results_file in os.listdir(activ_dir):
                if 'pt' not in results_file: continue
                results_filepath = f'{activ_dir}{results_file}'
                with h5py.File(results_filepath, 'r') as results:
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
                    n_data = clean_index.size 
                    activ = activ.reshape((n_data, -1))
                    for i, a in zip(clean_index, activ):
                        i = int(i)
                        utterance_prototypes[i] += a
                        utterance_count[i] += 1
                del activ
                gc.collect()
    if shuffle: # Undo set seed
        np.random.seed()

    # Calculate prototypes
    utterances_to_use = []
    for i, count in enumerate(utterance_count):
        utterance_prototypes[i] /= count
        if count > 0:
            utterances_to_use.append(i)
        else:
            print(f'Warning: utterance {i} contains no samples and will be skipped.')

    # Calculate and save utterance prototypes
    utterance_prototypes = np.array(utterance_prototypes)
    utterance_prototypes = utterance_prototypes[utterances_to_use, :]
    prototype_results = {
        'utterances': utterances_to_use, 'utterance_count': utterance_count,
        'prototypes': utterance_prototypes}
    prototypes_filename = f'utterance_prototypes_conv{conv_idx}_t{t}'
    if shuffle:
        prototypes_filename += f'_shuffle{shuffle_seed}'
    with open(f'{pca_activations_dir}{prototypes_filename}.p', 'wb') as f:
        pickle.dump(prototype_results, f, protocol=4)

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
        for t in [0]:
            print(f'====== PROCESSING LAYER {conv_idx}, TIMESTEP {t} ======')
            get_data_and_fit_PCA(conv_idx, t, pca_activations_dir, shuffle=shuffle)
            gc.collect()

