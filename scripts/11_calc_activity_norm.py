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
from models.pbranchednetwork_all import PBranchedNetwork_AllSeparateHP
from data.ValidationDataset import NoisyDataset, FullNoisyDataset

# Args
activations_string = str(sys.argv[1])
pnet_name = str(sys.argv[2])
chckpt = int(sys.argv[3])
tf_string = str(sys.argv[4])
if len(sys.argv) > 5:
    device_num = sys.argv[5]
    my_env = os.environ
    my_env["CUDA_VISIBLE_DEVICES"] = device_num

# Relevant paths and parameters
batch_size = 16
n_batches_to_test = 15
snrs = [-9.0, -6.0, -3.0, 0.0, 3.0]
bgs = ['Babble8Spkr', 'AudScene', 'pinkNoise']
PNetClass = PBranchedNetwork_AllSeparateHP
n_timesteps = 5
layers = ['conv1', 'conv2', 'conv3', 'conv4_W', 'conv5_W', 'fc6_W']
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}1_checkpoints/'
hyp_dir = f'{engram_dir}2_hyperp/'
activations_dir = f'{engram_dir}3_validation_activations/{activations_string}/'
pickles_dir = f'{engram_dir}pickles/'
main_tf_dir = f'{hyp_dir}{tf_string}/'

##### HELPER FUNCTIONS #####
def get_hyperparams(tf_filepath, bg, snr):
    hyperparams = []
    ea = event_accumulator.EventAccumulator(tf_filepath)
    ea.Reload()
    eval_score = [0]
    epoch = 0
    while True:
        try:
            score_over_t = 0.
            for t in np.arange(1,5):
                score_over_t += ea.Scalars(f'NoisyPerf/Epoch#{epoch}')[t].value
                epoch += 1
            score_over_t /= 4
            eval_score.append(score_over_t)
        except Exception as e:
            break
    for i in range(1, 6):
        hps = {}
        ffm = ea.Scalars(f'Hyperparam/pcoder{i}_feedforward')[-1].value
        fbm = ea.Scalars(f'Hyperparam/pcoder{i}_feedback')[-1].value
        erm = ea.Scalars(f'Hyperparam/pcoder{i}_error')[-1].value
        if np.isnan(ffm) or np.isnan(fbm) or np.isnan(erm):
            return None, 0.
        hps['ffm'] = ffm
        hps['fbm'] = fbm
        hps['erm'] = erm
        hyperparams.append(hps)
    return hyperparams, eval_score[-1]

def load_pnet(PNetClass, pnet_name, chckpt, hyperparams=None):
    net = BranchedNetwork()
    net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
    pnet = PNetClass(net, build_graph=False, track_forward_terms=True)
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

##### MAIN FUNCTION CALL #####
results = {
    'bg': [], 'snr': [], 'layer': [], 't': [],
    'ff': [], 'fb': [], 'mem': [], 'err': [],}
for bg in bgs:
    for snr in snrs:
        tf_dir = f'{main_tf_dir}hyper_{bg}_snr{snr}/'
        if not os.path.isdir(tf_dir): continue
        activ_dir = f'{activations_dir}{bg}_snr{int(snr)}/'
        os.makedirs(activ_dir, exist_ok=True)
        best_score = 0.
        best_hyperparams = None
        best_tf_file = None
        for tf_file in os.listdir(tf_dir):
            tf_filepath = f'{tf_dir}{tf_file}'
            tf_file = tf_file.split('edu.')[-1]
            hyperparams, score = get_hyperparams(tf_filepath, bg, snr)
            if hyperparams is None:
                continue
            if score > best_score:
                best_score = score
                best_hyperparams = hyperparams
                best_tf_file = tf_file
        print(f'{bg}, SNR {snr} uses {best_tf_file} with valid score {best_score}')

        # Use the best hyperparameter set
        pnet = load_pnet(PNetClass, pnet_name, chckpt, best_hyperparams)
        dset = NoisyDataset(bg, snr)
        dset_loader = DataLoader(
            dset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True)
        for batch_index, (images, _) in enumerate(dset_loader):
            if batch_index == n_batches_to_test:
                break
            pnet.reset()
            images = images.to(DEVICE)
            for t in range(n_timesteps):
                _input = images if t==0 else None
                outputs = pnet(_input)
                for layer in [1,2,3,4,5]:
                    if t == 0: break
                    pcoder = getattr(pnet, f'pcoder{layer}')
                    results['bg'].extend([bg]*images.shape[0])
                    results['snr'].extend([snr]*images.shape[0])
                    results['layer'].extend([layer]*images.shape[0])
                    results['t'].extend([t]*images.shape[0])
                    results['ff'].extend(pcoder.ff_norm.cpu().numpy().tolist())
                    results['fb'].extend(pcoder.fb_norm.cpu().numpy().tolist())
                    results['mem'].extend(pcoder.mem_norm.cpu().numpy().tolist())
                    results['err'].extend(pcoder.err_norm.cpu().numpy().tolist())

# Save results into a pickle file
os.makedirs(pickles_dir, exist_ok=True)
pfile = 'activity_norm.p'
pfile = f'{pickles_dir}{pnet_name}_{pfile}'
with open(pfile, 'wb') as f:
    pickle.dump(results, f)

