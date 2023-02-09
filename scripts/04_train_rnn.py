import os
import sys
import numpy as np
import socket
import gc
import h5py
import torch
from contextlib import closing
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from predify.utils.training import train_pcoders, eval_pcoders
from models.networks_2022 import BranchedNetwork
from data.ReconstructionTrainingDataset import NoisySoundsDataset
from models.pbranchednetwork_rnn import PBranchedNetwork_RNN
import torch.multiprocessing as mp
import torch.distributed as dist

# Fixed parameters, like directory paths
BATCH_SIZE = int(32/4)
MAX_TIMESTEP = 5
lr = 1E-5
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
checkpoints_dir = f'{engram_dir}1_checkpoints/'
tensorboard_dir = f'{engram_dir}1_tensorboard/'

def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # Initialize the process group.
    dist.init_process_group('NCCL', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def evaluate(
    net, epoch, test_dataset, timesteps,
    loss_function, writer=None, tag='Eval'):

    test_loss = np.zeros((timesteps+1,))
    correct = np.zeros((timesteps+1,))
    subset_size = 3000
    for _ in range(subset_size):
        idx = torch.randint(len(test_dataset), (1,))
        images, labels = test_dataset[idx]
        net.reset()
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            for tt in range(timesteps+1):
                if tt == 0:
                    outputs, _ = net(images)
                else:
                    outputs, _ = net(None)
                loss = loss_function(outputs, labels)
                test_loss[tt] += loss.item()
                _, preds = outputs.max(1)
                correct[tt] += preds.eq(labels).sum()
    print()
    for tt in range(timesteps+1):
        test_loss[tt] /= subset_size
        correct[tt] /= subset_size
        print('Test set t = {:02d}: Mean loss: {:.4f}, Accuracy: {:.4f}'.format(
            tt, test_loss[tt], correct[tt]))
        if writer is not None:
            writer.add_scalar(f"{tag}Perf/Epoch#{epoch}", correct[tt], tt)
    print()

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
            sumwriter.add_scalar(
                f"Hyperparam/pcoder{i}_memory",
                1 - getattr(net,f'ffm{i}').item() - getattr(net,f'fbm{i}').item(),
                epoch)
        else:
            sumwriter.add_scalar(
                f"Hyperparam/pcoder{i}_memory",
                1 - getattr(net,f'ffm{i}').item(),
                epoch)

def train_and_eval(gpu, args):
    # Unpack args
    load_pnet_name = args['load_pnet_name']
    load_pnet_chckpt = args['load_pnet_chckpt']
    num_epochs = args['num_epochs']
    num_gpus = args['num_gpus']
    pnet_name = args['pnet_name']
    free_port = args['free_port']

    # GPU set up
    print(f"USING GPU {gpu}")
    setup(gpu, num_gpus, free_port)
    #dist.init_process_group(
    #    backend='nccl', init_method='env://',
    #    world_size=num_gpus, rank=0)
    torch.cuda.set_device(gpu)

    # Set up net wrapper with correct hyperparameter with gradient
    net = BranchedNetwork()
    net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))
    pnet = PBranchedNetwork_RNN(net, build_graph=True)
    if load_pnet_chckpt > 0:
        print(f'LOADING network {load_pnet_name}-{load_pnet_chckpt}')
        state_dict_path = f'{checkpoints_dir}/{load_pnet_name}/{load_pnet_name}'
        state_dict_path += f'-{load_pnet_chckpt}-regular.pth'
        pnet.load_state_dict(torch.load(state_dict_path))
    pnet.eval()
    pnet.cuda(gpu)

    # Distributed data parallel
    ddp_pnet = torch.nn.parallel.DistributedDataParallel(
        pnet, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True)
    
    # Set up optimizer
    params_to_train = []
    trainable_parameter_names = pnet.get_trainable_parameter_names()
    trainable_parameter_names = ['module.' + n for n in trainable_parameter_names]
    for name, param in ddp_pnet.named_parameters():
        if name in trainable_parameter_names:
            param.requires_grad = True
            params_to_train.append(param)
        else:
            param.requires_grad = False
    loss_function = torch.nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(params_to_train, lr=lr, weight_decay=5E-4)

    # Set up dataset
    _datafile = 'hyperparameter_pooled_training_dataset_random_order_noNulls'
    train_dataset = NoisySoundsDataset(f'{engram_dir}{_datafile}.hdf5', subset=0.9)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=num_gpus, rank=gpu)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, sampler=train_sampler)
    if gpu == 0:
        test_dataset = NoisySoundsDataset(
            f'{engram_dir}{_datafile}.hdf5', subset=0.9, train=False)
    
    # Set up checkpoints and tensorboard
    if gpu == 0:
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
    for epoch in range(start_idx+1, num_epochs+start_idx+1):
        for batch_index, (images, labels) in enumerate(train_loader):
            pnet.reset()
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            ttloss = np.zeros((MAX_TIMESTEP+1))
            optimizer.zero_grad()
    
            for tt in range(MAX_TIMESTEP+1):
                if tt == 0:
                    outputs, _ = ddp_pnet(images)
                    loss = loss_function(outputs, labels)
                    ttloss[tt] = loss.item()
                else:
                    outputs, _ = ddp_pnet(None)
                    current_loss = loss_function(outputs, labels)
                    ttloss[tt] = current_loss.item()
                    loss = loss + current_loss
            loss.backward()
            optimizer.step()
            pnet.update_hyperparameters()

            if gpu == 0:
                progress_string = f'Training Epoch: {epoch} '
                progress_string += f'[{batch_index * BATCH_SIZE + len(images)}/'
                progress_string += f'{len(train_loader.dataset)}]'
                progress_string += f'Loss: {loss.item():0.4f}\t'
                progress_string += f'LR: {optimizer.param_groups[0]["lr"]:0.6f}'
                print(progress_string)
                for tt in range(MAX_TIMESTEP+1):
                    print(f'{ttloss[tt]:0.4f}\t', end='')
                hps = pnet.get_hyperparameters_values()
                print(f'\n{hps}\n')
                curr_batch = (epoch-1)*len(train_loader) + batch_index
                sumwriter.add_scalar(f"TrainingLoss/CE", loss.item(), curr_batch)
        if gpu == 0:
            log_hyper_parameters(pnet, epoch, sumwriter)
            if epoch % 5 == 0:
                torch.save(
                    pnet.state_dict(),
                    checkpoint_path.format(epoch=epoch, type='regular'))
                evaluate(
                    pnet, epoch, test_dataset, MAX_TIMESTEP,
                    loss_function, sumwriter)
    cleanup()

if __name__ == '__main__':
    # User-defined parameters
    task_id = int(sys.argv[1])
    load_pnet_name = str(sys.argv[2])
    load_pnet_chckpt = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
    num_gpus = int(sys.argv[5])
    pnet_name = f'pnet_rnn_{task_id}'
    free_port = get_open_port()

    # Collect into args
    args = {}
    args['load_pnet_name'] = load_pnet_name
    args['load_pnet_chckpt'] = load_pnet_chckpt
    args['num_epochs'] = num_epochs
    args['num_gpus'] = num_gpus
    args['pnet_name'] = pnet_name
    args['free_port'] = free_port
    
    mp.spawn(train_and_eval, nprocs=num_gpus, args=(args,))

