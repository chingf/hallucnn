import os
import sys
import numpy as np
import gc
import h5py
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from predify.utils.training import train_pcoders, eval_pcoders
from models.networks_2022 import BranchedNetwork
from data.ReconstructionTrainingDataset import CleanSoundsDataset
from data.ReconstructionTrainingDataset import NoisySoundsDataset
from models.pbranchednetwork_all import PBranchedNetwork_AllSeparateHP

# load user-defined parameters
pnet_name = str(sys.argv[1])
load_pnet_name = str(sys.argv[2])
load_pnet_chckpt = int(sys.argv[3])
NUM_EPOCHS = int(sys.argv[4])

# Set up PNet and dataset parameters
_train_datafile = 'clean_reconstruction_training_set'
SoundsDataset = CleanSoundsDataset
dset_kwargs = {'cgram_shuffle':2}
PNetClass = PBranchedNetwork_AllSeparateHP
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
BATCH_SIZE = 50
NUM_WORKERS = 2
PIN_MEMORY = True
lr = 1E-5

# Make directories
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}1_checkpoints/'
tensorboard_dir = f'{engram_dir}1_tensorboard/'
train_datafile = f'{engram_dir}{_train_datafile}.hdf5'

# Set up nets and optimizer
net = BranchedNetwork()
net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
pnet = PNetClass(net, build_graph=True)
if load_pnet_chckpt > 0:
    print(f'LOADING network {load_pnet_name}-{load_pnet_chckpt}')
    pnet.load_state_dict(torch.load(
        f'{checkpoints_dir}/{load_pnet_name}/{load_pnet_name}-{load_pnet_chckpt}-regular.pth'
        ))
pnet.eval()
pnet.to(DEVICE)
optimizer = torch.optim.Adam([
    {'params':getattr(pnet,f"pcoder{x+1}").pmodule.parameters(), 'lr':lr} for x in range(pnet.number_of_pcoders)
    ], weight_decay=5e-4)

# Set up dataset
train_dataset = SoundsDataset(train_datafile, subset=.9, **dset_kwargs)
test_dataset = SoundsDataset(
    train_datafile, subset=.9, train = False, **dset_kwargs
    )
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
for epoch in range(1, NUM_EPOCHS+1):
    train_pcoders(pnet, optimizer, loss_function, epoch, train_loader, DEVICE, sumwriter)
    eval_pcoders(pnet, loss_function, epoch, eval_loader, DEVICE, sumwriter)
    if epoch % 5 == 0:
        torch.save(
            pnet.state_dict(),
            checkpoint_path.format(epoch=epoch, type='regular')
            )
