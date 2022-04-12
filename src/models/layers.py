import numpy as np
import warnings
from collections import OrderedDict
from math import ceil

import torch
import torch.nn as nn
import pickle



class ConvLayer(nn.Module):
    """
    2D convolution and ReLU activation. NOTE: torch Conv2D uses NCHW format
    while tf conv2d uses NHWC format.
    Additionally, the padding choices are quite different between tf and torch:
    https://gist.github.com/Yangqing/47772de7eb3d5dbbff50ffb0d7a98964
    """

    def __init__(
            self, input_size, in_channels, out_channels, kernel_size, stride,
            printme=False, layer1 = False, layer2 = False
            ):

        super(ConvLayer, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.convcount = 1
        self.p0a,self.p0b = self._get_padding(input_size[0], kernel_size[0], stride) # H
        self.p1a, self.p1b = self._get_padding(input_size[1], kernel_size[1], stride) # W

        conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride,
                padding_mode='zeros', padding=(self.p0a, self.p1a),
                bias=True)
        self.block = nn.Sequential(conv, nn.ReLU())
        self.printme = printme

    def forward(self, _input):
        # NHWC to NCHW reshaping
        _input = torch.permute(_input, (0, 3, 1, 2))

        input_size = _input.size()
        
        # Make convolutional layer

        # Janky work around for the rectagular filters in conv1 
        if self.layer1:
            _input = torch.nn.functional.pad(_input, ( 0, 2, 0,0)) # [left, right, top, bot]
        if self.layer2: 
            _input = torch.nn.functional.pad(_input, ( 0, 0, 0,2)) # [left, right, top, bot]

        output = self.block(_input)
        # NCHW to NHWC reshaping
        output = torch.permute(output, (0, 2, 3, 1))

        """
        # Print Shape 
        print('conv ' + str(self.convcount) + ' layer:')
        self.convcount += 1 
        print(output.shape)
        """

        if self.printme:
            print('conv layer:')
            print(output.shape)
            print(output[0][0][0])
            print()
        return output

    def _get_padding(self, input_size, kernel_size, stride):
        """ Calculate manual padding in the style of tensorflow "SAME". """

        output_size = int(ceil(float(input_size) / float(stride)))
        pad_total = int((output_size - 1) * stride + kernel_size - input_size)
        pad_left = int(pad_total/2)
        pad_right = pad_total - pad_left
        
        if pad_left != pad_right:
            warnings.warn('Inconsistent tf pad calculation in ConvLayer.')
        return pad_left, pad_right

class LRNorm(nn.Module):
    """
    Local response normalization layer.
    tf uses alpha as scaling and torch uses alpha/n, where n is half the window.
    """

    def __init__(self, depth_radius, bias, alpha, beta, printme=False):
        super(LRNorm, self).__init__()
        self.block = nn.LocalResponseNorm(
            depth_radius*2+1, alpha*(depth_radius*2+1), beta, bias
            )
        self.lrnormcount = 1
        self.printme = printme

    def forward(self, _input):
        # NHWC to NCHW reshaping
        _input = torch.permute(_input, (0, 3, 1, 2))
        output = self.block(_input)
        # NCHW to NHWC reshaping
        output = torch.permute(output, (0, 2, 3, 1))
       
        """
        # Print Shape 
        print('lrnorm ' + str(self.lrnormcount) + ' layer:')
        self.lrnormcount += 1 
        print(output.shape)
        """

        if self.printme:
            print('norm layer:')
            print(output.shape)
            print(output[0][0][0])
            print()
        return output

class PoolLayer(nn.Module):
    """
    Pooling layer.
    Padding is unclear between tf and torch.
    tf: H_out = ceil(H_in/stride)
    torch: H_out = floor((H_in + 2*padding - kernel_size)/stride + 1)
    Watch out for padding and count_include_pad in torch.
    """

    def __init__(
            self, kernel_size, stride, max_pool=True, printme=False, input_size=None, layer1 = False
            ):
        super(PoolLayer, self).__init__()
        self.poolcount = 1
        self.layer1 = layer1
        if input_size is not None:
            padding = self._get_padding(input_size, kernel_size, stride)
        else:
            padding = int(kernel_size/2)
        if max_pool:
            self.block = nn.MaxPool2d(
                kernel_size, stride, padding=padding, ceil_mode=True
                )
        else:
            self.block = nn.AvgPool2d(
                kernel_size, stride, padding=padding, ceil_mode=True, count_include_pad=False
                )
        self.printme = printme

    def forward(self, _input):
        # NHWC to NCHW reshaping
        _input = torch.permute(_input, (0, 3, 1, 2))
        if self.layer1:
             _input = torch.nn.functional.pad(_input, ( 0, 0, 1,0)) # [left, right, top, bot]
        output = self.block(_input)
        # NCHW to NHWC reshaping
        output = torch.permute(output, (0, 2, 3, 1))

        """
        # Print Shape 
        print('pool' + str(self.poolcount) + ' layer:')
        self.poolcount += 1 
        print(output.shape)
        """

        if self.printme:
            print('pool layer:')
            print(output.shape)
            print(output[0][0][0])
            print()
         
        return output

    def _get_padding(self, input_size, kernel_size, stride):
        """ Calculate manual padding in the style of tensorflow "SAME". """

        output_size = int(ceil(float(input_size) / float(stride)))
        pad_total = int((output_size - 1) * stride + kernel_size - input_size)
        pad_left = int(pad_total/2)
        pad_right = pad_total - pad_left
        if pad_left != pad_right:
            warnings.warn(f'Inconsistent tf pad calculation: {pad_left}, {pad_right}')
        return pad_left

class FlattenPoolLayer(nn.Module):
    """ Flattens a pooling layer. """

    def __init__(self, output_size):
        super(FlattenPoolLayer, self).__init__()
        self.output_size = output_size

    def forward(self, _input):
        return torch.reshape(_input, (-1, self.output_size))

class FullyConnected(nn.Module):
    """ Fully connected layer with ReLU activation. """

    def __init__(self, input_size, output_size, activ=True):
        super(FullyConnected, self).__init__()
        if activ:
            self.block = nn.Sequential(
                nn.Linear(input_size, output_size), nn.ReLU()
                )
        else:
            self.block = nn.Linear(input_size, output_size)

    def forward(self, _input):
        return self.block(_input)

