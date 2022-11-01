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

task_number = int(sys.argv[1])

# Args
task_args = []
for noise_type in [
        ('auditory_scene', 'AudScene'),
        ('babble_8spkr', 'Babble8Spkr'),
        ('pink_noise', 'pinkNoise')
        ]:
    for snr_level in [('neg9', -9.)]:
        task_args.append((noise_type, snr_level))

# Set up dataset parameters
(bg_string, bg), (snr_string, snr) = task_args[task_number]
pnet_name = f'pnet_snr{int(snr)}_{bg}'
_train_datafile = 'hyperparameter_pooled_training_dataset_random_order_noNulls'
SoundsDataset = NoisySoundsDataset
dset_kwargs = {'snr': snr_string, 'bg': bg_string}

# Set up PNet parameters
PNetClass = PBranchedNetwork_AllSeparateHP

# Declare parameters
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
BATCH_SIZE = 50
NUM_WORKERS = 2
PIN_MEMORY = True
NUM_EPOCHS = 70
lr = 1E-5

# Make directories
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}checkpoints/'
tensorboard_dir = f'{engram_dir}tensorboard/'
train_datafile = f'{engram_dir}{_train_datafile}.hdf5'

# Set up nets and optimizer
net = BranchedNetwork()
net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
pnet = PNetClass(net, build_graph=True)
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
