import os
import sys
import numpy as np
import socket
import gc
import h5py
import torch
import time
from contextlib import closing
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from predify.utils.training import train_pcoders, eval_pcoders
from models.networks_2022 import BranchedNetwork
from models.pbranchednetwork_all import PBranchedNetwork_AllSeparateHP
from data.ReconstructionTrainingDataset import NoisySoundsDataset
import torch.multiprocessing as mp
import torch.distributed as dist


# # Global configurations

# Main args
MAX_TIMESTEP = 5
lr = 1E-4
NUM_WORKERS = 2

# Other training params
EPOCH = 100
MAX_TIMESTEP = 5

# # Helper functions

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

def load_pnet(
        net, state_dict, build_graph, random_init,
        ff_multiplier, fb_multiplier, er_multiplier, gpu):
    
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
    pnet.cuda(gpu)
    return pnet

def evaluate(
    ddpnet, net, epoch, dataloader, timesteps,
    loss_function, writer=None, tag='Clean'):

    test_loss = np.zeros((timesteps+1,))
    correct = np.zeros((timesteps+1,))
    for batch_index, (images, labels) in enumerate(dataloader):
        net.reset()
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.no_grad():
            for tt in range(timesteps+1):
                if tt == 0:
                    outputs, _ = ddpnet(images)
                else:
                    outputs, _ = ddpnet(None)
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

def train(
        ddpnet, net, epoch, dataloader, timesteps,
        loss_function, optimizer, gpu, writer):
    optimizer.zero_grad()
    for batch_index, (images, labels) in enumerate(dataloader):
        start_time = time.time()
        net.reset()
        labels = labels.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        ttloss = np.zeros((timesteps+1))
        #optimizer.zero_grad()

        for tt in range(timesteps+1):
            if tt == 0:
                outputs, _ = ddpnet(images)
                loss = loss_function(outputs, labels)
                ttloss[tt] = loss.item()
            else:
                outputs, _ = ddpnet(None)
                current_loss = loss_function(outputs, labels)
                ttloss[tt] = current_loss.item()
                loss = loss + current_loss
        if batch_index % N_BATCH_ACCUMULATE == 0:
            loss.backward()
            optimizer.step()
            net.update_hyperparameters()
            end_time = time.time()
            if (writer is not None) and (gpu==0):
                progress_string = f'training epoch: {epoch} '
                progress_string += f'[{batch_index * BATCH_SIZE + len(images)}/'
                progress_string += f'{len(dataloader.dataset)}]'
                progress_string += f'loss: {loss.item():0.4f}\t'
                progress_string += f'lr: {optimizer.param_groups[0]["lr"]:0.6f}'
                print(progress_string)
                for tt in range(MAX_TIMESTEP+1):
                    print(f'{ttloss[tt]:0.4f}\t', end='')
                hps = net.get_hyperparameters_values()
                print(f'\n{hps}\n')
                print(f'PROCESSING TIME: {end_time-start_time}')
                curr_batch = (epoch-1)*len(dataloader) + batch_index
                writer.add_scalar(f"TrainingLoss/CE", loss.item(), curr_batch)
            optimizer.zero_grad()
    if batch_index % N_BATCH_ACCUMULATE != 0:
        loss.backward()
        optimizer.step()
        net.update_hyperparameters()
# # Main hyperparameter optimization script

def train_and_eval(gpu, args):
    # Unpack args
    pnet_name = args['pnet_name']
    pnet_chckpt = args['pnet_chckpt']
    tensorboard_pnet_name = args['tensorboard_pnet_name']
    num_gpus = args['num_gpus']
    ablate = args['ablate']
    free_port = args['free_port']
    engram_dir = args['engram_dir']
    tensorboard_dir = args['tensorboard_dir']
    fb_state_dict_path = args['fb_state_dict_path']
    cuda_device = torch.device('cuda', gpu)
    fb_state_dict = torch.load(fb_state_dict_path, map_location=cuda_device)

    # GPU set up
    print(f"USING GPU {gpu}")
    setup(gpu, num_gpus, free_port)
    torch.cuda.set_device(gpu)

    # Set up net wrapper with correct hyperparameter with gradient
    net = BranchedNetwork()
    net.load_state_dict(
        torch.load(f'{engram_dir}networks_2022_weights.pt',
        map_location=cuda_device))
    if ablate == None:
        ffm = np.random.uniform()
        fbm = np.random.uniform(high=1.-ffm)
        erm = np.random.uniform()*0.1
    elif ablate == 'erm':
        fbm = np.random.uniform()
        ffm = np.random.uniform(high=1.-fbm)
        erm = 0.
    else: # fbm ablation
        fbm = 0.
        ffm = np.random.uniform(high=1.-fbm)
        erm = np.random.uniform()
    pnet = load_pnet(
        net, fb_state_dict, build_graph=True, random_init=False,
        ff_multiplier=ffm, fb_multiplier=fbm, er_multiplier=erm, gpu=gpu)

    # Distributed data parallel
    ddp_pnet = torch.nn.parallel.DistributedDataParallel(
        pnet, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True)

    # Set up optimizer
    params_to_train = []
    for name, param in ddp_pnet.named_parameters():
        if 'ff_part' in name:
            param.requires_grad = True
            params_to_train.append(param)
        elif ('fb_part' in name) and (ablate != 'fbm'):
            param.requires_grad = True
            params_to_train.append(param)
        elif 'mem_part' in name:
            param.requires_grad = True
            params_to_train.append(param)
        elif ('errorm' in name) and (ablate != 'erm'):
            param.requires_grad = True
            params_to_train.append(param)
        else:
            param.requires_grad = False
    loss_function = torch.nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(params_to_train, lr=lr, weight_decay=0.00001)

    # Set up noisy data
    _datafile = 'hyperparameter_pooled_training_dataset_random_order_noNulls'
    train_dataset = NoisySoundsDataset(f'{engram_dir}{_datafile}.hdf5', subset=0.9)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=num_gpus, rank=gpu)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=True, sampler=train_sampler, drop_last=False)
    if gpu == 0:
        test_dataset = NoisySoundsDataset(
            f'{engram_dir}{_datafile}.hdf5', subset=0.9, train=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            pin_memory=True, drop_last=False)

    # Set up tensorboard and log initial hyperparameters/validation score
    if gpu == 0:
        os.makedirs(tensorboard_dir, exist_ok=True)
        sumwriter = SummaryWriter(tensorboard_dir)
        log_hyper_parameters(pnet, 0, sumwriter)
        hps = pnet.get_hyperparameters_values()
        print('Hyperparameters:')
        print(hps)
        evaluate(
            ddp_pnet, pnet, 0, test_loader, MAX_TIMESTEP, loss_function,
            writer=sumwriter, tag='Noisy')

    # Iterate through epochs
    for epoch in range(1, EPOCH+1):
        writer = sumwriter if gpu==0 else None
        train(
            ddp_pnet, pnet, epoch, train_loader, MAX_TIMESTEP,
            loss_function, optimizer, gpu, writer)
        if gpu == 0:
            log_hyper_parameters(pnet, epoch, sumwriter)
            hps = pnet.get_hyperparameters_values()
            print('Hyperparameters:')
            print(hps)
            evaluate(
                ddp_pnet, pnet, epoch, test_loader, MAX_TIMESTEP, loss_function,
                writer=sumwriter, tag='Noisy')
    cleanup()
    if gpu == 0:
        sumwriter.close()

if __name__ == '__main__':
    task_number = int(sys.argv[1])
    pnet_name = str(sys.argv[2])
    pnet_chckpt = int(sys.argv[3])
    tensorboard_pnet_name = str(sys.argv[4])
    num_gpus = int(sys.argv[5]) # 4
    BATCH_SIZE = int(sys.argv[6]) # 6
    N_BATCH_ACCUMULATE = int(sys.argv[7]) #4
    if len(sys.argv) > 8:
        print("Ablation argument received.")
        ablate = str(sys.argv[8])
        if (ablate != 'erm') and (ablate != 'fbm'):
            raise ValueError('Not valid ablation type')
    else:
        ablate = None
    free_port = get_open_port()
    
    # Set up necessary paths
    engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'
    checkpoints_dir = f'{engram_dir}1_checkpoints/'
    tensorboard_dir = f'{engram_dir}2_hyperp/{tensorboard_pnet_name}/hyper_all/'
    fb_state_dict_path = f'{checkpoints_dir}{pnet_name}/{pnet_name}-{pnet_chckpt}-regular.pth'

    print("=====================")
    print(f'All noise types')
    print(tensorboard_dir)
    print("=====================")

    # Collect into args
    args = {}
    args['pnet_name'] = pnet_name
    args['pnet_chckpt'] = pnet_chckpt
    args['tensorboard_pnet_name'] = tensorboard_pnet_name
    args['num_gpus'] = num_gpus
    args['ablate'] = ablate
    args['free_port'] = free_port
    args['engram_dir'] = engram_dir
    args['tensorboard_dir'] = tensorboard_dir
    args['fb_state_dict_path'] = fb_state_dict_path

    mp.spawn(train_and_eval, nprocs=num_gpus, args=(args,))

