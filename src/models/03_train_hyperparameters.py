import os
import sys
import numpy as np
import gc
import h5py
root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from predify.utils.training import train_pcoders, eval_pcoders

from networks_2022 import BranchedNetwork
from data.CleanSoundsDataset import CleanSoundsDataset
from data.NoisyDataset import NoisyDataset, FullNoisyDataset, LargeNoisyDataset

task_number = int(sys.argv[1])

# # Global configurations

# Args
task_args = []
for noise_types in [['AudScene'], ['Babble8Spkr'], ['pinkNoise']]:
    for snr_levels in [[-9.], [-6.], [-3.], [0.], [3.]]:
        task_args.append((noise_types, snr_levels))

# Main args
task_number = task_number % len(task_args)
noise_types, snr_levels = task_args[task_number]
LR_SCALING = 0.01
SAME_PARAM = False

# Dataset configuration
BATCH_SIZE = 10
NUM_WORKERS = 2

# Other training params
EPOCH = 100
FF_START = True             # to start from feedforward initialization
MAX_TIMESTEP = 5


# Path names
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}checkpoints/'
tensorboard_dir = f'{engram_dir}tensorboard/randomInit_lr_{LR_SCALING}x/'

# # Load network arguments

if SAME_PARAM:
    from pbranchednetwork_shared import PBranchedNetwork_SharedSameHP
    PNetClass = PBranchedNetwork_SharedSameHP
    pnet_name = 'pnet'
    fb_state_dict_path = f'{checkpoints_dir}{pnet_name}/{pnet_name}-shared-50-regular.pth'
else:
    from pbranchednetwork_all import PBranchedNetwork_AllSeparateHP
    PNetClass = PBranchedNetwork_AllSeparateHP
    pnet_name = 'pnet'
    fb_state_dict_path = f'{checkpoints_dir}{pnet_name}/{pnet_name}-50-regular.pth'
fb_state_dict = torch.load(fb_state_dict_path)


# # Helper functions

def load_pnet(
        net, state_dict, build_graph, random_init,
        ff_multiplier, fb_multiplier, er_multiplier,
        same_param, device='cuda:0'):
    
    pnet = PNetClass(
        net, build_graph=build_graph, random_init=random_init,
        ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier, er_multiplier=er_multiplier
        )

    pnet.load_state_dict(state_dict)
    hyperparams = []
    for i in range(1, 6):
        ffm = np.random.uniform()
        fbm = np.random.uniform(high=1.-ffm)
        erm = np.random.uniform()*0.1
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


def train_and_eval(noise_type, snr_level):
    # Load clean and noisy data
    clean_ds_path = f'{engram_dir}training_dataset_random_order.hdf5'
    clean_ds = CleanSoundsDataset(clean_ds_path)
    clean_loader = torch.utils.data.DataLoader(
        clean_ds,  batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False, num_workers=NUM_WORKERS
        )

    noisy_ds = LargeNoisyDataset(bg=noise_type, snr=snr_level)
    noise_loader = torch.utils.data.DataLoader(
        noisy_ds,  batch_size=BATCH_SIZE,
        shuffle=True, drop_last=False,
        num_workers=NUM_WORKERS
        )

    # Set up logs and network for training
    net_dir = f'hyper_{noise_type}_snr{snr_level}'
    if not FF_START:
        net_dir += '_randomInit'
    if SAME_PARAM:
        net_dir += '_shared'

    sumwriter = SummaryWriter(f'{tensorboard_dir}{net_dir}')
    net = BranchedNetwork() # Load original network
    net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
    pnet_fw = load_pnet( # Load FF PNet
        net, fb_state_dict, build_graph=False, random_init=(not FF_START),
        ff_multiplier=1.0, fb_multiplier=0.0, er_multiplier=0.0,
        same_param=SAME_PARAM, device='cuda:0'
        )
    loss_function = torch.nn.CrossEntropyLoss()
    evaluate(
        pnet_fw, 0, noise_loader, 1,
        loss_function,
        writer=sumwriter, tag='FeedForward')
    del pnet_fw
    gc.collect()

    # Load PNet for hyperparameter optimization
    net = BranchedNetwork()
    net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
    ffm = np.random.uniform()
    fbm = np.random.uniform()
    erm = np.random.uniform()*0.1
    pnet = load_pnet(
        net, fb_state_dict, build_graph=True, random_init=(not FF_START),
        ff_multiplier=ffm, fb_multiplier=fbm, er_multiplier=erm,
        same_param=SAME_PARAM, device='cuda:0'
        )

    # Set up loss function and hyperparameters
    loss_function = torch.nn.CrossEntropyLoss()
    hyperparams = [*pnet.get_hyperparameters()]
    if SAME_PARAM:
        optimizer = torch.optim.Adam([
            {'params': hyperparams[:-1], 'lr':0.01*LR_SCALING},
            {'params': hyperparams[-1:], 'lr':0.0001*LR_SCALING}], weight_decay=0.00001)
    else:
        fffbmem_hp = []
        erm_hp = []
        for pc in range(pnet.number_of_pcoders):
            fffbmem_hp.extend(hyperparams[pc*4:pc*4+3])
            erm_hp.append(hyperparams[pc*4+3])
        optimizer = torch.optim.Adam([
            {'params': fffbmem_hp, 'lr':0.01*LR_SCALING},
            {'params': erm_hp, 'lr':0.0001*LR_SCALING}], weight_decay=0.00001)

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
            pnet, epoch, noise_loader,
            MAX_TIMESTEP, loss_function, optimizer,
            writer=sumwriter
            )
        log_hyper_parameters(pnet, epoch, sumwriter, same_param=SAME_PARAM)
        hps = pnet.get_hyperparameters_values()
        print(hps)

        evaluate(
            pnet, epoch, noise_loader,
            MAX_TIMESTEP, loss_function,
            writer=sumwriter, tag='Noisy'
            )
    sumwriter.close()


for noise_type in noise_types:
    for snr_level in snr_levels:
        for _ in range(5):
            print("=====================")
            print(f'{noise_type}, for SNR {snr_level}')
            print(tensorboard_dir)
            print("=====================")
            train_and_eval(noise_type, snr_level)


