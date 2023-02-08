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
from data.ReconstructionTrainingDataset import NoisySoundsDataset
from models.pbranchednetwork_rnn import PBranchedNetwork_RNN

# User-defined parameters
TASK_ID = int(sys.argv[1])
load_pnet_name = str(sys.argv[2])
load_pnet_chckpt = int(sys.argv[3])
NUM_EPOCHS = int(sys.argv[4])
pnet_name = f'pnet_rnn_{TASK_ID}'


# Default parameters
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
BATCH_SIZE = 32
NUM_WORKERS = 1
MAX_TIMESTEP = 5
PIN_MEMORY = True
lr = 1E-5

# Set up directory paths
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}1_checkpoints/'
tensorboard_dir = f'{engram_dir}1_tensorboard/'

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
        hps = net.get_hyperparameters_values()
        print(f'\n{hps}')
        print()
        if writer is not None:
            writer.add_scalar(
                f"TrainingLoss/CE", loss.item(),
                (epoch-1)*len(dataloader) + batch_index)

def evaluate(net, epoch, dataloader, timesteps, loss_function, writer=None, tag='Eval'):
    test_loss = np.zeros((timesteps+1,))
    correct = np.zeros((timesteps+1,))
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
            tt, test_loss[tt], correct[tt]))
        if writer is not None:
            writer.add_scalar(f"{tag}Perf/Epoch#{epoch}", correct[tt], tt)
    print()

def log_hyper_parameters(net, epoch, sumwriter):
    for i in range(1, net.number_of_pcoders+1):
        sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedforward", getattr(net,f'ffm{i}').item(), epoch)
        if i < net.number_of_pcoders:
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedback", getattr(net,f'fbm{i}').item(), epoch)
        else:
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedback", 0, epoch)
        sumwriter.add_scalar(f"Hyperparam/pcoder{i}_error", getattr(net,f'erm{i}').item(), epoch)
        if i < net.number_of_pcoders:
            sumwriter.add_scalar(
                f"Hyperparam/pcoder{i}_memory",
                1 - getattr(net,f'ffm{i}').item() - getattr(net,f'fbm{i}').item(),
                epoch)
        else:
            sumwriter.add_scalar(
                f"Hyperparam/pcoder{i}_memory",
                1 - getattr(net,f'ffm{i}').item(),
                epoch)

def train_and_eval():
    # Set up net wrapper with correct hyperparameter with gradient
    net = BranchedNetwork()
    net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
    pnet = PBranchedNetwork_RNN(net, build_graph=True)
    if load_pnet_chckpt > 0:
        print(f'LOADING network {load_pnet_name}-{load_pnet_chckpt}')
        pnet.load_state_dict(torch.load(
            f'{checkpoints_dir}/{load_pnet_name}/{load_pnet_name}-{load_pnet_chckpt}-regular.pth'
            ))
    pnet.eval()
    pnet.to(DEVICE)
    
    # Set up optimizer # TODO
    loss_function = torch.nn.CrossEntropyLoss()
    for param in pnet.parameters(): # Turn off grad for everything
        param.requires_grad = False
    trainable_params = pnet.get_trainable_parameters(lr, lr)
    for param_dict in trainable_params: # Only feedback and time constant
        for param in param_dict['params']:
            param.requires_grad = True
    optimizer = torch.optim.Adam(trainable_params, weight_decay=5E-4)
    
    # Set up dataset
    _datafile = 'hyperparameter_pooled_training_dataset_random_order_noNulls'
    train_dataset = NoisySoundsDataset(f'{engram_dir}{_datafile}.hdf5', subset=0.9)
    test_dataset = NoisySoundsDataset(
        f'{engram_dir}{_datafile}.hdf5', subset=0.9, train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    eval_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # Set up checkpoints and tensorboard
    checkpoint_path = os.path.join(checkpoints_dir, f"{pnet_name}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, pnet_name + '-{epoch}-{type}.pth')
    tensorboard_path = os.path.join(tensorboard_dir, f"{pnet_name}")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    sumwriter = SummaryWriter(tensorboard_path, filename_suffix=f'')
    
    # Iterate through epochs
    start_idx = 0
    if (load_pnet_chckpt) > 0 and (pnet_name == load_pnet_name):
        start_idx = load_pnet_chckpt
    for epoch in range(start_idx+1, NUM_EPOCHS+start_idx+1):
        train(pnet, epoch, train_loader, MAX_TIMESTEP,
            loss_function, optimizer, writer=sumwriter)
        log_hyper_parameters(pnet, epoch, sumwriter)
        evaluate(pnet, epoch, eval_loader, MAX_TIMESTEP, loss_function, sumwriter)
        if epoch % 5 == 0:
            torch.save(
                pnet.state_dict(),
                checkpoint_path.format(epoch=epoch, type='regular'))

train_and_eval()
