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
from data.HyperparameterTrainingDataset import NoisySoundsDataset

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

task_args = []
for bg in ['AudScene', 'Babble8Spkr', 'pinkNoise']:
    for snr in [-9., -6., -3., 0., 3.]:
        task_args.append((bg, snr))
bg, snr = task_args[task_number]

# # Global configurations

# Dataset configuration
HOSTNAME = os.environ['HOSTNAME']
if HOSTNAME in ['ax11', 'ax12', 'ax13', 'ax14', 'ax15', 'ax16']:
    print(f'Host is {HOSTNAME} and running A40 parameters')
    BATCH_SIZE = 48
    N_BATCH_ACCUMULATE = 2
else:
    print(f'Host is {HOSTNAME} and running 1080 parameters')
    BATCH_SIZE = 6
    N_BATCH_ACCUMULATE = 16
NUM_WORKERS = 2
LR_SCALING = 0.01 * (6/(BATCH_SIZE*N_BATCH_ACCUMULATE)) 
LR_SCALING = LR_SCALING * 100 # TODO grid search

# Other training params
EPOCH = 200
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
        ff_multiplier, fb_multiplier, er_multiplier, device='cuda:0'):
    
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

        if batch_index % N_BATCH_ACCUMULATE == 0:
            loss.backward()
            optimizer.step()
            net.update_hyperparameters()
            progress_string = f'training epoch: {epoch} '
            progress_string += f'[{batch_index * BATCH_SIZE + len(images)}/'
            progress_string += f'{len(dataloader.dataset)}]'
            progress_string += f'loss: {loss.item():0.4f}\t'
            progress_string += f'lr: {optimizer.param_groups[0]["lr"]:0.6f}'           
            print(progress_string)
            for tt in range(timesteps+1):
                print(f'{ttloss[tt]:0.4f}\t', end='')
            if writer is not None:
                writer.add_scalar(
                    f"TrainingLoss/CE", loss.item(),
                    (epoch-1)*len(dataloader) + batch_index)

            optimizer.zero_grad()
    if batch_index % N_BATCH_ACCUMULATE != 0:
        loss.backward()
        optimizer.step()
        net.update_hyperparameters()

def log_hyper_parameters(net, epoch, sumwriter):
    for i in range(1, net.number_of_pcoders+1):
        sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedforward",
            getattr(net,f'ffm{i}').item(), epoch)
        if i < net.number_of_pcoders:
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedback",
                getattr(net,f'fbm{i}').item(), epoch)
        else:
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedback", 0, epoch)
        sumwriter.add_scalar(f"Hyperparam/pcoder{i}_error",
            getattr(net,f'erm{i}').item(), epoch)
        if i < net.number_of_pcoders:
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_memory",
                1-getattr(net,f'ffm{i}').item()-getattr(net,f'fbm{i}').item(), epoch)
        else:
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_memory",
                1-getattr(net,f'ffm{i}').item(), epoch)


def train_and_eval(bg, snr):
    # Load noisy data
    _datafile = 'hyperparameter_pooled_training_dataset_random_order_noNulls'
    noisy_ds = NoisySoundsDataset(
        f'{engram_dir}{_datafile}.hdf5', bg=bg, snr=snr)
    noise_loader = torch.utils.data.DataLoader(
        noisy_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        num_workers=NUM_WORKERS)
    net_dir = f'hyper_{bg}_snr{snr}'

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
        net, fb_state_dict, build_graph=True, random_init=False,
        ff_multiplier=ffm, fb_multiplier=fbm, er_multiplier=erm, device='cuda:0')

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
            {'params': erm_hp, 'lr':0.01*LR_SCALING}
            ], weight_decay=0.00001)

    # Log initial hyperparameter and eval values
    log_hyper_parameters(pnet, 0, sumwriter)
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
        log_hyper_parameters(pnet, epoch, sumwriter)
        hps = pnet.get_hyperparameters_values()
        print(hps)

        evaluate(
            pnet, epoch, noise_loader,
            MAX_TIMESTEP, loss_function,
            writer=sumwriter, tag='Noisy'
            )
    sumwriter.close()

net_dir = f'hyper_{bg}_snr{snr}'
net_dir = f'{tensorboard_dir}{net_dir}'
os.makedirs(net_dir, exist_ok=True)
#n_tboards = len(os.listdir(net_dir))
#n_iters = max(0, max_iters-n_tboards)
print("=====================")
print(f'{bg}, for SNR {snr}')
print(tensorboard_dir)
print("=====================")

for _ in range(4):
    train_and_eval(bg, snr)
    
