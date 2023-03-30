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

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from predify.utils.training import train_pcoders, eval_pcoders

from models.networks_2022 import BranchedNetwork
from data.ReconstructionTrainingDataset import CleanSoundsDataset
from data.NoisyDataset import NoisyDataset, FullNoisyDataset

# Batch params
task_number = int(sys.argv[1])
activations_string = str(sys.argv[2])
pnet_name = str(sys.argv[3])
chckpt = int(sys.argv[4])
tf_string = str(sys.argv[5])
merged_hyperparameters = int(sys.argv[6])
if len(sys.argv) > 7:
    device_num = sys.argv[7]
    my_env = os.environ
    my_env["CUDA_VISIBLE_DEVICES"] = device_num

# ARG LIST
task_args = []
for snrs in [[-9.0], [-6.0], [-3.0], [0.0], [3.0]]:
    for bgs in [['Babble8Spkr'], ['AudScene'], ['pinkNoise']]:
        task_args.append((snrs, bgs))
task_number = task_number % len(task_args)
snrs, bgs = task_args[task_number]

# PNET PARAMETERS
from models.pbranchednetwork_all import PBranchedNetwork_AllSeparateHP
PNetClass = PBranchedNetwork_AllSeparateHP
n_timesteps = 5
layers = ['conv1', 'conv2', 'conv3', 'conv4_W', 'conv5_W', 'fc6_W']

# # Paths to relevant directories
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}1_checkpoints/'
hyp_dir = f'{engram_dir}2_hyperp/'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# WHICH MODELS?
activations_dir = f'{engram_dir}3_activations/{activations_string}/'
main_tf_dir = f'{hyp_dir}{tf_string}/'

# # Helper functions to load network

def get_hyperparams(tf_filepath, bg, snr):
    hyperparams = []
    ea = event_accumulator.EventAccumulator(tf_filepath)
    ea.Reload()
    #try:
    #    _eval_acc = ea.Scalars(f'NoisyPerf/Epoch#80')[0].value
    #except:
    #    return None
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
    activations = []
    logits = []
    output = []
    for t in range(n_timesteps):
        _input_t = _input if t == 0 else None
        logits_t, _ = pnet(_input_t)
        activations.append(pnet.backbone.encoder_repr)
        logits.append(logits_t.cpu().numpy().squeeze())
        output.append(logits_t.max(-1)[1].item())
    return activations, logits, output

@torch.no_grad()
def save_activations(pnet, dset, hdf5_path):
    
    with h5py.File(hdf5_path, 'x') as f_out:
        data_dict = {}
        data_dict['label'] = f_out.create_dataset(
            'label', dset.n_data, dtype='float32'
            )
        data_dict['clean_correct'] = f_out.create_dataset(
            'clean_correct', dset.n_data, dtype='float32'
            )
        data_dict['pnet_correct'] = f_out.create_dataset(
            'pnet_correct', dset.n_data, dtype='float32'
            )
        for layer_idx, layer in enumerate(layers):
            activ_dim = (dset.n_data,) + n_units_per_layer[layer]
            logit_dim = (dset.n_data, 531)
            for timestep in range(n_timesteps):
                data_dict[f'{layer}_{timestep}_activations'] = f_out.create_dataset(
                    f'{layer}_{timestep}_activations', activ_dim, dtype='float32'
                    )
                data_dict[f'{layer}_{timestep}_clean_activations'] = f_out.create_dataset(
                    f'{layer}_{timestep}_clean_activations', activ_dim, dtype='float32'
                    )
                if layer_idx == 0:
                    data_dict[f'{timestep}_logits'] = f_out.create_dataset(
                        f'{timestep}_logits', logit_dim, dtype='float32'
                        )
                    data_dict[f'{timestep}_output'] = f_out.create_dataset(
                        f'{timestep}_output', dset.n_data, dtype='float32'
                        )
                    data_dict[f'{timestep}_clean_logits'] = f_out.create_dataset(
                        f'{timestep}_clean_logits', logit_dim, dtype='float32'
                        )
                    data_dict[f'{timestep}_clean_output'] = f_out.create_dataset(
                        f'{timestep}_clean_output', dset.n_data, dtype='float32'
                        )
    
        for idx in range(dset.n_data):
            # Noisy input
            noisy_in, label = dset[idx]
            data_dict['label'][idx] = label
            noisy_in = noisy_in.to(DEVICE)
            activations, logits, output = run_pnet(pnet, noisy_in)
            data_dict['pnet_correct'][idx] = label == output[-1]
            for timestep in range(n_timesteps):
                for layer in layers:
                    data_dict[f'{layer}_{timestep}_activations'][idx] = \
                        activations[timestep][layer]
                data_dict[f'{timestep}_logits'][idx] = \
                    logits[timestep]
                data_dict[f'{timestep}_output'][idx] = output[timestep]

            # Clean input
            clean_in = torch.tensor(
                dset.clean_in[idx].reshape((1, 1, 164, 400))
                ).to(DEVICE)
            clean_in = clean_in.to(DEVICE)
            activations, logits, output = run_pnet(pnet, clean_in)
            data_dict['clean_correct'][idx] = label == output[0]
            for timestep in range(n_timesteps):
                for layer in layers:
                    data_dict[f'{layer}_{timestep}_clean_activations'][idx] = \
                        activations[timestep][layer]
                data_dict[f'{timestep}_clean_logits'][idx] = \
                    logits[timestep]
                data_dict[f'{timestep}_clean_output'][idx] = output[timestep]
                    

# # Run activation-saving functions
for bg in bgs:
    for snr in snrs:
        if merged_hyperparameters == 0:
            tf_dir = f'{main_tf_dir}hyper_{bg}_snr{snr}/'
        else:
            tf_dir = f'{main_tf_dir}hyper_all/'
        if not os.path.isdir(tf_dir): continue
        activ_dir = f'{activations_dir}{bg}_snr{int(snr)}/'
        os.makedirs(activ_dir, exist_ok=True)
        for tf_file in os.listdir(tf_dir):
            tf_filepath = f'{tf_dir}{tf_file}'
            tf_file = tf_file.split('edu.')[-1]
            hyperparams = get_hyperparams(tf_filepath, bg, snr)
            if hyperparams is None:
                continue
            pnet = load_pnet(PNetClass, pnet_name, chckpt, hyperparams)
            dset = NoisyDataset(bg, snr)
            hdf5_path = f'{activ_dir}{tf_file}.hdf5'
            if os.path.exists(hdf5_path):
                continue
            save_activations(pnet, dset, hdf5_path)

