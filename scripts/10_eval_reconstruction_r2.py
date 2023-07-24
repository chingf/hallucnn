#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import sys
import numpy as np
import gc
import pickle
import time
import h5py
from tensorboard.backend.event_processing import event_accumulator
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from predify.utils.training import train_pcoders, eval_pcoders

from models.networks_2022 import BranchedNetwork
from models.pbranchednetwork_all import PBranchedNetwork_AllSeparateHP
from data.ValidationDataset import NoisyDataset, FullNoisyDataset, CleanCounterpart
from utils import eval_pcoders_r2

# Arguments 
pnet_name = str(sys.argv[1])
chckpt = int(sys.argv[2])
clean_data = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    device_num = sys.argv[4]
    my_env = os.environ
    my_env["CUDA_VISIBLE_DEVICES"] = device_num

# Relevant paths and parameters
snrs = [3.0, 0.0, -3.0, -6.0, -9.0]
bgs = ['Babble8Spkr', 'AudScene', 'pinkNoise']
PNetClass = PBranchedNetwork_AllSeparateHP
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}1_checkpoints/'
hyp_dir = f'{engram_dir}2_hyperp/'
pickles_dir = f'{engram_dir}pickles/'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# Helper functions
def load_pnet(PNetClass, pnet_name, chckpt, hyperparams=None):
    net = BranchedNetwork()
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

def main():
    results = {'bg': [], 'snr': [], 'score': [], 'layer': []}
    for bg in bgs:
        for snr in snrs:
            # Use the best hyperparameter set
            pnet = load_pnet(PNetClass, pnet_name, chckpt)
            if clean_data:
                dset = CleanCounterpart(bg, snr)
            else:
                dset = NoisyDataset(bg, snr)
            eval_loader = DataLoader(
                dset, batch_size=16, shuffle=False,
                num_workers=2, pin_memory=True)
            scores = eval_pcoders_r2(pnet, eval_loader, DEVICE)
            for layer, score in enumerate(scores):
                results['bg'].append(bg)
                results['snr'].append(snr)
                results['score'].append(score)
                results['layer'].append(layer)
            del pnet
            del dset
            del eval_loader
            gc.collect()

            if clean_data:
                break
        if clean_data:
            break

    # Save results into a pickle file
    os.makedirs(pickles_dir, exist_ok=True)
    if clean_data:
        pfile = 'reconstruction_r2_clean.p'
    else:
        pfile = 'reconstruction_r2.p'
    pfile = f'{pickles_dir}{pnet_name}_{pfile}'
    with open(pfile, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    elapsed_time = (end-start)/60.
    print(f'ELAPSED TIME: {elapsed_time} minutes')

