import os
import sys
import re
import numpy as np
import gc
import h5py
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from utils import train_pcoders, eval_pcoders
from models.networks_2022 import BranchedNetwork
from data.ReconstructionTrainingDataset import CleanSoundsDataset
from data.HyperparameterTrainingDataset import TmpNoisyDatasetForReconstructionTraining
from models.pbranchednetwork_all import PBranchedNetwork_AllSeparateHP

# load user-defined parameters
pnet_name = str(sys.argv[1])
NUM_EPOCHS = int(sys.argv[2])
if len(sys.argv) > 3:
    dset_mod = str(sys.argv[3])
else:
    dset_mod = None

# Set up PNet and dataset parameters
if dset_mod == None:
    print('Reconstruction on clean sounds')
    _train_datafile = 'clean_reconstruction_training_set'
    SoundsDataset = CleanSoundsDataset
    dset_kwargs = {}
elif dset_mod == 'temp_shuffle':
    print('Reconstruction on temporally-shuffled sounds')
    _train_datafile = 'clean_reconstruction_training_set'
    SoundsDataset = CleanSoundsDataset
    dset_kwargs = {'cgram_shuffle':2}
elif dset_mod == 'freq_shuffle':
    print('Reconstruction on frequency-shuffled sounds')
    _train_datafile = 'clean_reconstruction_training_set'
    SoundsDataset = CleanSoundsDataset
    dset_kwargs = {'cgram_shuffle':1}
elif dset_mod == 'noisy':
    print('Reconstruction on noisy sounds')
    _train_datafile = 'hyperparameter_pooled_training_dataset_random_order_noNulls'
    SoundsDataset = TmpNoisyDatasetForReconstructionTraining
    dset_kwargs = {}
else:
    raise ValueError('Unrecognized dataset modification')
PNetClass = PBranchedNetwork_AllSeparateHP
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
NUM_WORKERS = 2
PIN_MEMORY = True

HOSTNAME = os.environ['HOSTNAME']
if HOSTNAME in ['ax11', 'ax12', 'ax13', 'ax14', 'ax15', 'ax16']:
    print(f'Host is {HOSTNAME} and running A40 parameters')
    BATCH_SIZE = 208
    N_BATCH_ACCUMULATE = 1
else:
    print(f'Host is {HOSTNAME} and running 1080 parameters')
    BATCH_SIZE = 34
    N_BATCH_ACCUMULATE = 6
lr = 1E-5 * (50/(BATCH_SIZE*N_BATCH_ACCUMULATE))

# Make directories
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}1_checkpoints/'
tensorboard_dir = f'{engram_dir}1_tensorboard/'
train_datafile = f'{engram_dir}{_train_datafile}.hdf5'

# Set up nets and optimizer
net = BranchedNetwork()
net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
pnet = PNetClass(net, build_graph=True)

# Load from checkpoints
start_epoch = 0
if os.path.isdir(f'{checkpoints_dir}{pnet_name}/'):
    chckpt_epochs = [0]
    regex = f'{pnet_name}-(\d+)-regular\.pth'
    for chckpt_file in os.listdir(f'{checkpoints_dir}{pnet_name}/'):
        m = re.search(regex, chckpt_file)
        chckpt_epochs.append(int(m.group(1)))
    max_chckpt_epoch = max(chckpt_epochs)
    if max_chckpt_epoch > 0:
        start_epoch = max_chckpt_epoch
        print(f'LOADING network {pnet_name}-{start_epoch}')
        pnet.load_state_dict(torch.load(
            f'{checkpoints_dir}{pnet_name}/{pnet_name}-{start_epoch}-regular.pth'
            ))
pnet.eval()
pnet.to(DEVICE)
optimizer = torch.optim.Adam([
    {'params':getattr(pnet,f"pcoder{x+1}").pmodule.parameters(), 'lr':lr} for x in range(pnet.number_of_pcoders)
    ], weight_decay=5e-4)

# Set up dataset
train_dataset = SoundsDataset(train_datafile, train=True, **dset_kwargs)
test_dataset = SoundsDataset(train_datafile, train=False, **dset_kwargs)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
eval_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

# Set up checkpoints and tensorboard
checkpoint_path = os.path.join(checkpoints_dir, f"{pnet_name}")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, pnet_name + '-{epoch}-{type}.pth')
tensorboard_path = os.path.join(tensorboard_dir, f"{pnet_name}")
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
sumwriter = SummaryWriter(tensorboard_path, filename_suffix=f'')

# Train
loss_function = torch.nn.MSELoss()
for epoch in range(start_epoch, NUM_EPOCHS+1):
    train_pcoders(
        pnet, optimizer, loss_function, epoch, train_loader, DEVICE, sumwriter,
        n_batch_accumulate=N_BATCH_ACCUMULATE)
    eval_pcoders(pnet, loss_function, epoch, eval_loader, DEVICE, sumwriter)
    if epoch % 5 == 0:
        torch.save(
            pnet.state_dict(),
            checkpoint_path.format(epoch=epoch, type='regular')
            )
