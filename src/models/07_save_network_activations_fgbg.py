#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import seaborn as sns
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from scipy.stats import pearsonr
root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from predify.utils.training import train_pcoders, eval_pcoders

from networks_2022 import BranchedNetwork
from data.CleanSoundsDataset import CleanSoundsDataset
from data.NoisyDataset import NoisyDataset, FullNoisyDataset

# Batch params
task_number = int(sys.argv[1])
task_args = []
for snrs in [[-9.0], [-6.0], [-3.0], [0.0], [3.0]]:
    for bgs in [['Babble8Spkr'], ['AudScene'], ['pinkNoise']]:
        task_args.append((snrs, bgs))
task_number = task_number % len(task_args)
snrs, bgs = task_args[task_number]

# # PNet parameters

from pbranchednetwork_all import PBranchedNetwork_AllSeparateHP
PNetClass = PBranchedNetwork_AllSeparateHP
pnet_name = 'pnet'
chckpt = 50
n_timesteps = 5
layers = ['conv1', 'conv2', 'conv3', 'conv4_W', 'conv5_W', 'fc6_W']

# # Paths to relevant directories
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
activations_dir = f'{engram_dir}activations_pnet/'
checkpoints_dir = f'{engram_dir}checkpoints/'
tensorboard_dir = f'{engram_dir}tensorboard/'
main_tf_dir = f'{tensorboard_dir}randomInit_lr_0.01x/'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')


# # Helper functions to load network

def get_hyperparams(tf_filepath, bg, snr, shared=False):
    if shared:
        raise ValueError('Not implemented for shared hyperparameters.')
        
    hyperparams = []
    ea = event_accumulator.EventAccumulator(tf_filepath)
    ea.Reload()
    try:
        _eval_acc = ea.Scalars(f'NoisyPerf/Epoch#80')[0].value
    except:
        return None
    for i in range(1, 6):
        hps = {}
        ffm = ea.Scalars(f'Hyperparam/pcoder{i}_feedforward')[-1].value
        fbm = ea.Scalars(f'Hyperparam/pcoder{i}_feedback')[-1].value
        erm = ea.Scalars(f'Hyperparam/pcoder{i}_error')[-1].value
        if np.isnan(ffm) or np.isnan(fbm) or np.isnan(erm):
            return None
        hps['ffm'] = ffm
        hps['fbm'] = fbm
        hps['erm'] = erm
        hyperparams.append(hps)
    return hyperparams

def load_pnet(PNetClass, pnet_name, chckpt, hyperparams=None):
    net = BranchedNetwork(track_encoder_representations=True)
    net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
    pnet = PNetClass(net, build_graph=False)
    pnet.load_state_dict(torch.load(
        f"{checkpoints_dir}{pnet_name}/{pnet_name}-{chckpt}-regular.pth",
        map_location='cpu'
        ))
    if hyperparams is not None:
        pnet.set_hyperparameters(hyperparams)
    pnet.to(DEVICE)
    pnet.eval();
    print(f'Loaded Pnet: {pnet_name}')
    print_hps(pnet)
    return pnet

def print_hps(pnet):
    for pc in range(pnet.number_of_pcoders):
        print (f"PCoder{pc+1} : ffm: {getattr(pnet,f'ffm{pc+1}'):0.3f} \t fbm: {getattr(pnet,f'fbm{pc+1}'):0.3f} \t erm: {getattr(pnet,f'erm{pc+1}'):0.3f}")


# # Helper functions to save activations

n_units_per_layer = {
    'conv1': (96, 55, 134), 'conv2': (256, 14, 34),
    'conv3': (512, 7, 17), 'conv4_W': (1024, 7, 17),
    'conv5_W': (512, 7, 17), 'fc6_W': (4096,)
    }

def run_pnet(pnet, _input):
    pnet.reset()
    reconstructions = []
    activations = []
    logits = []
    output = []
    for t in range(n_timesteps):
        _input_t = _input if t == 0 else None
        logits_t, _ = pnet(_input_t)
        reconstructions.append(pnet.pcoder1.prd[0,0].cpu().numpy())
        activations.append(pnet.backbone.encoder_repr)
        logits.append(logits_t.cpu().numpy().squeeze())
        output.append(logits_t.max(-1)[1].item())
    return reconstructions, activations, logits, output

@torch.no_grad()
def save_activations(pnet, data, hdf5_path):
    n_data = data.shape[0]
    
    with h5py.File(hdf5_path, 'x') as f_out:
        data_dict = {}
        for layer_idx, layer in enumerate(layers):
            activ_dim = (n_data,) + n_units_per_layer[layer]
            for timestep in range(n_timesteps):
                data_dict[f'{layer}_{timestep}_activations'] = f_out.create_dataset(
                    f'{layer}_{timestep}_activations', activ_dim, dtype='float32'
                    )
        for idx in range(data.shape[0]):
            # Noisy input
            noisy_in = data[idx]
            reconstructions, activations, logits, output = run_pnet(pnet, noisy_in)
            for timestep in range(n_timesteps):
                for layer in layers:
                    data_dict[f'{layer}_{timestep}_activations'][idx] = \
                        activations[timestep][layer]


# # Run activation-saving functions
files = ['fg', 'bg_july22']
tf_dir = f'{tensorboard_dir}lr_0.01x/'

for bg in bgs:
    for snr in snrs:
        tf_dir = f'{main_tf_dir}hyper_{bg}_snr{snr}/'
        for tf_file in os.listdir(tf_dir):
            tf_filepath = f'{tf_dir}{tf_file}'
            tf_file = tf_file.split('edu.')[-1]
            hyperparams = get_hyperparams(tf_filepath, bg, snr)
            if hyperparams is None:
                continue
            pnet = load_pnet(PNetClass, pnet_name, chckpt, hyperparams)
            for noise_file in files:
                hdf5_path = f'{activations_dir}{noise_file}/{tf_file}.hdf5'
                loaded_file = h5py.File(f'{engram_dir}{noise_file}.hdf5', 'r')
                data = np.array(loaded_file['data']).reshape((-1, 164, 400))
                data = torch.tensor(data).to(DEVICE)
                save_activations(pnet, data, hdf5_path)

