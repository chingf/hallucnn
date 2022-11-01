import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gamma
from models.networks_2022 import BranchedNetwork
from models.pbranchednetwork_all import PBranchedNetwork_AllSeparateHP
from data.ReconstructionTrainingDataset import CleanSoundsDataset 

# Which network to test
pnet_name = 'pnet_gammaNoise'
chckpt = 70

# Set up parameters
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}checkpoints/'
tensorboard_dir = f'{engram_dir}tensorboard/'
datafile = f'{engram_dir}seed_542_word_clean_random_order.hdf5'
PNetClass = PBranchedNetwork_AllSeparateHP
n_timesteps = 5
layers = ['conv1', 'conv2', 'conv3', 'conv4_W', 'conv5_W', 'fc6_W']

# Load network
net = BranchedNetwork()
net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
pnet = PNetClass(net, build_graph=True)
def print_hps(pnet):
    for pc in range(pnet.number_of_pcoders):
        string = f"PCoder{pc+1} : ffm: {getattr(pnet,f'ffm{pc+1}'):0.3f} \t"
        string += f"fbm: {getattr(pnet,f'fbm{pc+1}'):0.3f} \t"
        string += f"erm: {getattr(pnet,f'erm{pc+1}'):0.3f}"
        print(string)
pnet.load_state_dict(torch.load(
    f"{checkpoints_dir}{pnet_name}/{pnet_name}-50-regular.pth",
    map_location='cpu'
    ))
pnet.to(DEVICE)
pnet.build_graph = False
pnet.eval();
print_hps(pnet)

# Set up test dataset
full_dataset = CleanSoundsDataset(datafile)
n_train = int(len(full_dataset)*0.9)
eval_dataset = Subset(full_dataset, np.arange(n_train, len(full_dataset)))
del full_dataset

# Plot reconstructions over successive timesteps
for i in range(5):
    pnet.reset()
    fig, axs = plt.subplots(5, 1, figsize = (6, 8))
    with torch.no_grad():
        for j in range(n_timesteps):
            _input = eval_dataset[i][0] if j == 0 else None
            if _input is not None:
                _input = _input.to(DEVICE)
                test_shape = _input.shape
            outputs = pnet(_input)
            reconstruction = np.array(pnet.pcoder1.prd[0,0].cpu())
            axs[j].imshow(reconstruction)
            axs[j].set_title(f'Timestep {j}')
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f'PCoder 1, {test_shape}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'../figures/16b_{pnet_name}_input(i).png', dpi=300)
