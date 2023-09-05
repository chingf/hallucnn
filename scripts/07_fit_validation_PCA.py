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
netname = str(sys.argv[1]) # pnet
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
activations_dir = f'{engram_dir}3_validation_activations/{netname}/'
pca_activations_dir = f'{engram_dir}4_validation_PCA/{netname}/'
bg_types = ['pinkNoise', 'AudScene', 'Babble8Spkr']
snr_types = [-9.0, -6.0, -3.0, 0.0, 3.0]
n_units_to_sample = 500
seeds = np.arange(25)

##### HELPER FUNCTIONS #####
def get_data_and_fit_PCA(bg, snr):
    # Load all the data for this BG/SNR pair
    save_dir = f'{pca_activations_dir}{bg}_snr{int(snr)}/'
    os.makedirs(save_dir, exist_ok=True)
    activ_dir = f'{activations_dir}{bg}_snr{int(snr)}/'
    for results_file in os.listdir(activ_dir):
        results_filepath = f'{activ_dir}{results_file}'
        results = h5py.File(results_filepath, 'r')

    # Iterate through convolutional layer and timestep
    for conv_idx in [1, 2, 3, 4, 5]:
        for t in [0, 1, 2, 3, 4]:
            if conv_idx > 3:
                activ = np.array(results[f'conv{conv_idx}_W_{t}_activations'])
            else:
                activ = np.array(results[f'conv{conv_idx}_{t}_activations'])
            n_data = activ.shape[0]
            activ = activ.reshape((n_data, -1))
            n_units = activ.shape[1]

            # Now run random subsamples:
            for seed in seeds:
                np.random.seed(seed)
                print(f'Layer {conv_idx}, timestep {t}, seed {seed}')
                idxs = np.random.choice(n_units, size=n_units_to_sample, replace=False)
                pca = PCA()
                pca.fit(activ[:,idxs])
                pca_filename = f'conv{conv_idx}_t{t}_seed{seed}'
                with open(f'{save_dir}{pca_filename}.p', 'wb') as f:
                    pickle.dump(pca, f, protocol=4)

def get_cpu_usage():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t --giga').readlines()[-1].split()[1:])

    # Memory usage
    p_used = round((used_memory/total_memory) * 100, 2)
    print(f"RAM {used_memory} GB, {p_used}% used")

##### MAIN CALL #####

if __name__ == "__main__":
    for bg in bg_types:
        for snr in snr_types:
            print(f'====== PROCESSING BG {bg}, SNR {snr} ======')
            get_data_and_fit_PCA(bg, snr)
            gc.collect()

