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
from models.pbranchednetwork_all import PBranchedNetwork_AllSeparateHP
#from data.NoisyDataset import NoisyDataset, FullNoisyDataset, LargeNoisyDataset
from data.ReconstructionTrainingDataset import NoisySoundsDataset

task_number = int(sys.argv[1])
pnet_name = str(sys.argv[2])
pnet_chckpt = int(sys.argv[3])
tensorboard_pnet_name = str(sys.argv[4])
if len(sys.argv) > 5:
    print("Ablation argument received.")
    ablate = str(sys.argv[5])
    if (ablate != 'erm') and (ablate != 'fbm'):
        raise ValueError('Not valid ablation type')
else:
    ablate = None

# # Global configurations

# Main args
LR_SCALING = 0.01
SAME_PARAM = False

# Dataset configuration
BATCH_SIZE = 50
NUM_WORKERS = 2

# Other training params
EPOCH = 100
FF_START = True             # to start from feedforward initialization
MAX_TIMESTEP = 5

# Path names
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}1_checkpoints/'
tensorboard_dir = f'{engram_dir}2_hyperp/{tensorboard_pnet_name}/'

# # Load network arguments
fb_state_dict_path = f'{checkpoints_dir}{pnet_name}/{pnet_name}-{pnet_chckpt}-regular.pth'
fb_state_dict = torch.load(fb_state_dict_path)

# # Helper functions

def load_pnet(
        net, state_dict, build_graph, random_init,
        ff_multiplier, fb_multiplier, er_multiplier,
        same_param, device='cuda:0'):
    
    pnet = PBranchedNetwork_AllSeparateHP(
        net, build_graph=build_graph, random_init=random_init,
        ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier, er_multiplier=er_multiplier
        )
    pnet.load_state_dict(state_dict)
    hyperparams = []
    for i in range(1, 6):
        hps = {}
        hps['ffm'] = ff_multiplier
        hps['fbm'] = fb_multiplier
        hps['erm'] = er_multiplier
        hyperparams.append(hps)
    pnet.set_hyperparameters(hyperparams)
    pnet.eval()
    pnet.to(device)
    return pnet


def evaluate(net, epoch, dataloader, timesteps, loss_function, writer=None, tag='Clean'):
    test_loss = np.zeros((timesteps+1,))
    correct   = np.zeros((timesteps+1,))
    for (images, labels) in dataloader:
        images = images.cuda()
        labels = labels.cuda()
        
        with torch.no_grad():
            for tt in range(timesteps+1):
                if tt == 0:
                    outputs, _ = net(images)
                else:
                    outputs, _ = net()
                
                loss = loss_function(outputs, labels)
                test_loss[tt] += loss.item()
                _, preds = outputs.max(1)
                correct[tt] += preds.eq(labels).sum()

    print()
    for tt in range(timesteps+1):
        test_loss[tt] /= len(dataloader.dataset)
        correct[tt] /= len(dataloader.dataset)
        print('Test set t = {:02d}: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            tt,
            test_loss[tt],
            correct[tt]
        ))
        if writer is not None:
            writer.add_scalar(
                f"{tag}Perf/Epoch#{epoch}",
                correct[tt], tt
                )
    print()


def train(net, epoch, dataloader, timesteps, loss_function, optimizer, writer=None):
    for batch_index, (images, labels) in enumerate(dataloader):
        net.reset()

        labels = labels.cuda()
        images = images.cuda()

        ttloss = np.zeros((timesteps+1))
        optimizer.zero_grad()

        for tt in range(timesteps+1):
            if tt == 0:
                outputs, _ = net(images)
                loss = loss_function(outputs, labels)
                ttloss[tt] = loss.item()
            else:
                outputs, _ = net()
                current_loss = loss_function(outputs, labels)
                ttloss[tt] = current_loss.item()
                #if tt == timesteps:# TODO: check timestep effect
                loss += current_loss
        
        loss.backward()
        optimizer.step()
        net.update_hyperparameters()
            
        print(f"Training Epoch: {epoch} [{batch_index * BATCH_SIZE + len(images)}/{len(dataloader.dataset)}]\tLoss: {loss.item():0.4f}\tLR: {optimizer.param_groups[0]['lr']:0.6f}")
        for tt in range(timesteps+1):
            print(f'{ttloss[tt]:0.4f}\t', end='')
        print()
        if writer is not None:
            writer.add_scalar(
                f"TrainingLoss/CE", loss.item(),
                (epoch-1)*len(dataloader) + batch_index
                )


def log_hyper_parameters(net, epoch, sumwriter, same_param=True):
    if same_param:
        sumwriter.add_scalar(f"HyperparamRaw/feedforward", getattr(net,f'ff_part').item(), epoch)
        sumwriter.add_scalar(f"HyperparamRaw/feedback",    getattr(net,f'fb_part').item(), epoch)
        sumwriter.add_scalar(f"HyperparamRaw/error",       getattr(net,f'errorm').item(), epoch)
        sumwriter.add_scalar(f"HyperparamRaw/memory",      getattr(net,f'mem_part').item(), epoch)

        sumwriter.add_scalar(f"Hyperparam/feedforward", getattr(net,f'ffm').item(), epoch)
        sumwriter.add_scalar(f"Hyperparam/feedback",    getattr(net,f'fbm').item(), epoch)
        sumwriter.add_scalar(f"Hyperparam/error",       getattr(net,f'erm').item(), epoch)
        sumwriter.add_scalar(f"Hyperparam/memory",      1-getattr(net,f'ffm').item()-getattr(net,f'fbm').item(), epoch)
    else:
        for i in range(1, net.number_of_pcoders+1):
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedforward", getattr(net,f'ffm{i}').item(), epoch)
            if i < net.number_of_pcoders:
                sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedback", getattr(net,f'fbm{i}').item(), epoch)
            else:
                sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedback", 0, epoch)
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_error", getattr(net,f'erm{i}').item(), epoch)
            if i < net.number_of_pcoders:
                sumwriter.add_scalar(f"Hyperparam/pcoder{i}_memory",      1-getattr(net,f'ffm{i}').item()-getattr(net,f'fbm{i}').item(), epoch)
            else:
                sumwriter.add_scalar(f"Hyperparam/pcoder{i}_memory",      1-getattr(net,f'ffm{i}').item(), epoch)


# # Main hyperparameter optimization script

# In[ ]:


def train_and_eval():
    # Load noisy data
    _datafile = 'hyperparameter_pooled_training_dataset_random_order_noNulls'
    noisy_ds = NoisySoundsDataset(f'{engram_dir}{_datafile}.hdf5')
    noise_loader = torch.utils.data.DataLoader(
        noisy_ds,  batch_size=BATCH_SIZE,
        shuffle=True, drop_last=False,
        num_workers=NUM_WORKERS
        )

    # Set up logs and network for training
    net_dir = f'hyper_all'
    if not FF_START:
        net_dir += '_randomInit'
    if SAME_PARAM:
        net_dir += '_shared'

    # Load PNet for hyperparameter optimization
    sumwriter = SummaryWriter(f'{tensorboard_dir}{net_dir}')
    net = BranchedNetwork()
    net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
    if ablate == None:
        ffm = np.random.uniform()
        fbm = np.random.uniform(high=1.-ffm)
        erm = np.random.uniform()*0.1
    elif ablate == 'erm':
        fbm = np.random.uniform()
        ffm = np.random.uniform(high=1.-fbm)
        erm = 0.
    else: # fb
        fbm = 0.
        ffm = np.random.uniform(high=1.-fbm)
        erm = np.random.uniform()
    pnet = load_pnet(
        net, fb_state_dict, build_graph=True, random_init=(not FF_START),
        ff_multiplier=ffm, fb_multiplier=fbm, er_multiplier=erm,
        same_param=SAME_PARAM, device='cuda:0'
        )

    # Set up loss function and hyperparameters
    loss_function = torch.nn.CrossEntropyLoss()
    for param in pnet.parameters():
        param.requires_grad = False
    hyperparams = [*pnet.get_hyperparameters()]
    for hyperparam in hyperparams:
        hyperparam.requires_grad = True
    if ablate == None:
        fffbmem_hp = []
        erm_hp = []
        for pc in range(pnet.number_of_pcoders):
            fffbmem_hp.extend(hyperparams[pc*4:pc*4+3])
            erm_hp.append(hyperparams[pc*4+3])
        optimizer = torch.optim.Adam([
            {'params': fffbmem_hp, 'lr':0.01*LR_SCALING},
            {'params': erm_hp, 'lr':0.0001*LR_SCALING}], weight_decay=0.00001)
    elif ablate == 'erm':
        fffbmem_hp = []
        for pc in range(pnet.number_of_pcoders):
            fffbmem_hp.extend(hyperparams[pc*4:pc*4+3])
        optimizer = torch.optim.Adam([
            {'params': fffbmem_hp, 'lr':0.01*LR_SCALING}
            ], weight_decay=0.00001)
    else: # fb
        ffmem_hp = []
        erm_hp = []
        for pc in range(pnet.number_of_pcoders):
            ffmem_hp.extend([hyperparams[pc*4], hyperparams[pc*4+2]])
            erm_hp.append(hyperparams[pc*4+3])
        optimizer = torch.optim.Adam([
            {'params': ffmem_hp, 'lr':0.01*LR_SCALING},
            {'params': erm_hp, 'lr':0.0001*LR_SCALING}
            ], weight_decay=0.00001)

    # Log initial hyperparameter and eval values
    log_hyper_parameters(pnet, 0, sumwriter, same_param=SAME_PARAM)
    hps = pnet.get_hyperparameters_values()
    print(hps)
    evaluate(
        pnet, 0, noise_loader,
        MAX_TIMESTEP, loss_function,
        writer=sumwriter, tag='Noisy'
        )

    # Run epochs
    for epoch in range(1, EPOCH+1):
        train(
            pnet, epoch, noise_loader, MAX_TIMESTEP,
            loss_function, optimizer, writer=sumwriter)
        log_hyper_parameters(pnet, epoch, sumwriter, same_param=SAME_PARAM)
        hps = pnet.get_hyperparameters_values()
        print(hps)

        evaluate(
            pnet, epoch, noise_loader,
            MAX_TIMESTEP, loss_function,
            writer=sumwriter, tag='Noisy'
            )
    sumwriter.close()

net_dir = 'hyper_all'
net_dir = f'{tensorboard_dir}{net_dir}'
os.makedirs(net_dir, exist_ok=True)
print("=====================")
print(f'All noise types')
print(tensorboard_dir)
print("=====================")
train_and_eval()

