# Copied from predify package to alter train_pcoders for batch accumulation

from typing import Callable
import torch
import numpy as np
from sklearn.metrics import r2_score

def train_pcoders(
    net: torch.nn.Module, optimizer: torch.optim.Optimizer,
    loss_function: Callable, epoch: int, train_loader: torch.utils.data.DataLoader,
    device: str, writer: torch.utils.tensorboard.SummaryWriter=None,
    n_batch_accumulate: int=1):
    r"""
    Trains the feedback modules of PCoders using a distance between the prediction of a PCoder and the
    representation of the PCoder below.

    Args:
        net (torch.nn.Module): Predified network including all the PCoders
        optimizer (torch.optim.Optimizer): PyTorch-compatible optimizer object
        loss_function (Callable): A callable function that receives two tensors
        and returns the distance between them
        epoch (int): Training epoch number
        train_loader (torch.utils.data.DataLoader): DataLoader for training samples
        writer (torch.utils.tensorboard.SummaryWrite, optional):
        Tensorboard summary writer to track training history. Default: None
        device (str): Training device (e.g. 'cpu', 'cuda:0')
    """
    
    net.train()
    net.backbone.eval()

    nb_trained_samples = 0
    optimizer.zero_grad()
    for batch_index, (images, _) in enumerate(train_loader):
        net.reset()
        images = images.to(device)
        outputs = net(images)
        for i in range(net.number_of_pcoders):
            if i == 0:
                a = loss_function(net.pcoder1.prd, images)
                loss = a
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                a = loss_function(pcoder_curr.prd, pcoder_pre.rep)
                loss += a
            if writer is not None:
                writer.add_scalar(f"MSE Train/PCoder{i+1}", a.item(), (epoch-1) * len(train_loader) + batch_index)
        
        nb_trained_samples += images.shape[0]

        if batch_index % n_batch_accumulate == 0: 
            loss.backward()
            optimizer.step()
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}'.format(
                loss.item(), epoch=epoch, trained_samples=nb_trained_samples,
                total_samples=len(train_loader.dataset)))
            if writer is not None:
                writer.add_scalar(f"MSE Train/Sum", loss.item(),
                    (epoch-1) * len(train_loader) + batch_index)
            optimizer.zero_grad()
    if batch_index % n_batch_accumulate != 0: 
        loss.backward()
        optimizer.step()
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}'.format(
            loss.item(), epoch=epoch, trained_samples=nb_trained_samples,
            total_samples=len(train_loader.dataset)))
        if writer is not None:
            writer.add_scalar(f"MSE Train/Sum", loss.item(),
                (epoch-1) * len(train_loader) + batch_index)
        optimizer.zero_grad()

@torch.no_grad()
def eval_pcoders(
    net: torch.nn.Module, loss_function: Callable, epoch: int,
    eval_loader: torch.utils.data.DataLoader, device: str,
    writer: torch.utils.tensorboard.SummaryWriter=None):
    r"""
    Evaluates the feedback modules of PCoders using a distance between the prediction of a PCoder and the
    representation of the PCoder below.

    Args:
        net (torch.nn.Module): Predified network including all the PCoders
        loss_function (Callable): A callable function that receives two tensors and returns the distance between them
        epoch (int): Evaluation epoch number
        test_loader (torch.utils.data.DataLoader): DataLoader for evaluation samples
        writer (torch.utils.tensorboard.SummaryWrite, optional): Tensorboard summary writer to track evaluation history. Default: None
        device (str): Training device (e.g. 'cpu', 'cuda:0')
    """

    net.eval()

    final_loss = [0 for i in range(net.number_of_pcoders)]
    for batch_index, (images, _) in enumerate(eval_loader):
        net.reset()
        images = images.to(device)
        outputs = net(images)
        for i in range(net.number_of_pcoders):
            if i == 0:
                final_loss[i] += loss_function(net.pcoder1.prd, images).item()
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                final_loss[i] += loss_function(pcoder_curr.prd, pcoder_pre.rep).item()
    
    loss_sum = 0
    for i in range(net.number_of_pcoders):
        final_loss[i] /= len(eval_loader)
        loss_sum += final_loss[i]
        if writer is not None:
            writer.add_scalar(f"MSE Eval/PCoder{i+1}", final_loss[i], epoch-1)
            
            
    print('Training Epoch: {epoch} [{evaluated_samples}/{total_samples}]\tLoss: {:0.4f}'.format(
        loss_sum,
        epoch=epoch,
        evaluated_samples=len(eval_loader.dataset),
        total_samples=len(eval_loader.dataset)
    ))
    if writer is not None:
        writer.add_scalar(f"MSE Eval/Sum", loss_sum, epoch-1)

@torch.no_grad()
def eval_pcoders_r2(
    net: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader, device: str,
    units_to_sample: list=[]):
    r"""
    Returns:
        scores: (n_pcoders,) array; each element is the r2 value, averaged over
        the D-dimensions of the pcoder.
    """

    net.eval()

    ss_residuals = [None for l in range(net.number_of_pcoders)]
    targets = [[] for l in range(net.number_of_pcoders)]
    mean_target = [None for l in range(net.number_of_pcoders)]
    n_images = 0
    for batch_index, (images, _) in enumerate(eval_loader):
        net.reset()
        images = images.to(device)
        outputs = net(images)
        n_images += images.shape[0]
        for l in range(net.number_of_pcoders):
            if l == 0:
                target = images.reshape(images.shape[0], -1)
                prd = net.pcoder1.prd.reshape(images.shape[0], -1)
            else:
                pcoder_pre = getattr(net, f"pcoder{l}").rep
                pcoder_curr = getattr(net, f"pcoder{l+1}").prd
                target = pcoder_pre.reshape(pcoder_pre.shape[0], -1)
                prd = pcoder_curr.reshape(pcoder_curr.shape[0], -1)

            # Sample units if needed
            if len(units_to_sample) > 0:
                target = target[:, units_to_sample[l]]
                prd = prd[:, units_to_sample[l]]

            resid = target - prd
            if ss_residuals[l] is None:
                ss_residuals[l] = np.zeros(resid.shape[1])
                mean_target[l] = np.zeros(resid.shape[1])
            ss_residuals[l] += torch.sum(torch.square(resid), dim=0).cpu().numpy()
            targets[l].extend([t.cpu().numpy() for t in target])
            mean_target[l] += torch.sum(target, dim=0).cpu().numpy()

    # Calculate mean_response
    mean_target = [t/n_images for t in mean_target]

    # Calculate sum-of-squares total
    ss_total = []
    for l in range(net.number_of_pcoders):
        _targets = np.array(targets[l])
        _mean_target = mean_target[l]
        _ss_total = np.sum(np.square(_targets - _mean_target[None,:]), axis=0)
        ss_total.append(_ss_total)

    # Calculate r^2 score
    scores = [ss_residuals[l] / ss_total[l] for l in range(net.number_of_pcoders)]
    scores = [1-r for r in scores]
    scores = [np.median(r) for r in scores]
    return scores
    
